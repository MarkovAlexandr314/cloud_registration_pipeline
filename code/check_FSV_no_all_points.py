import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import os


# free-space violation


# ============================================================
# ПАРАМЕТРЫ LIDAR LIVOX MID-360S
# ============================================================
LIVOX_H_FOV_DEG = 360.0          # горизонтальный FOV
LIVOX_V_FOV_MIN_DEG = -7.0       # нижняя граница вертикального FOV
LIVOX_V_FOV_MAX_DEG = 52.0       # верхняя граница вертикального FOV
LIVOX_MIN_RANGE_M = 0.1          # мёртвая зона
LIVOX_MAX_RANGE_M = 100.0        # cut-off range


def load_trajectory(filepath):
    """
    Загружает траекторию: timestamp x y z qx qy qz qw.
    Возвращает timestamps, positions (N,3), orientations (N,4) — кватернионы.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл траектории не найден: {filepath}")

    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    timestamps = data[:, 0]
    positions = data[:, 1:4]
    orientations = data[:, 4:]  # (qx, qy, qz, qw)

    print(f"[*] Траектория '{filepath}': {len(timestamps)} поз")
    return timestamps, positions, orientations


def evaluate_free_space_consistency(
    cloud_source,
    cloud_reference,
    trajectory_positions,
    trajectory_orientations,
    tolerance=0.2,
    collision_radius=0.2,
    surface_margin=0.6,
    surface_knn_check=True,
    surface_knn_radius=0.6,
    max_rays_per_pose=100,
    max_poses=None,
    pose_seed=42,
    ray_seed=42,
    max_pairs_per_pose=30,
    max_violation_pairs=5000,
):
    source_pts = np.asarray(cloud_source.points, dtype=np.float64)
    ref_pts = np.asarray(cloud_reference.points, dtype=np.float64)

    if len(source_pts) == 0 or len(ref_pts) == 0:
        print("[!] Одно из облаков пустое.")
        return None

    ref_tree = KDTree(ref_pts)
    source_tree = KDTree(source_pts)

    n_poses_total = len(trajectory_positions)
    if max_poses is not None and max_poses < n_poses_total:
        pose_rng = np.random.RandomState(pose_seed)
        sampled_idx = np.sort(pose_rng.choice(n_poses_total, max_poses, replace=False))
        print(f"[*] Подвыборка поз: {max_poses} из {n_poses_total} (seed={pose_seed})")
    else:
        sampled_idx = np.arange(n_poses_total)

    poses_use = trajectory_positions[sampled_idx]
    orients_use = trajectory_orientations[sampled_idx]

    half_v_min_rad = np.radians(LIVOX_V_FOV_MIN_DEG)
    half_v_max_rad = np.radians(LIVOX_V_FOV_MAX_DEG)

    ray_rng = np.random.RandomState(ray_seed)

    total_rays_checked = 0
    total_contradictions = 0
    skipped_by_surface = 0
    skipped_by_surface_knn = 0
    bad_ref_idx_set = set()
    violation_pairs = []

    # ← НОВОЕ: сколько точек-нарушителей отброшено как «не первая на луче»
    skipped_not_first = 0

    print(f"[*] Проверка {len(sampled_idx)} поз")
    print(f"[*] FOV: H=360°, V=[{LIVOX_V_FOV_MIN_DEG},{LIVOX_V_FOV_MAX_DEG}]°")
    print(f"[*] tolerance={tolerance}, collision_radius={collision_radius}, "
          f"surface_margin={surface_margin}, surface_knn_radius={surface_knn_radius}")

    for local_i, P in enumerate(poses_use):
        q = orients_use[local_i]
        try:
            rot = Rotation.from_quat(q)
            rot_inv = rot.inv()
        except ValueError:
            continue

        local_pts = rot_inv.apply(source_pts - P)
        x_l, y_l, z_l = local_pts[:, 0], local_pts[:, 1], local_pts[:, 2]
        distances = np.linalg.norm(local_pts, axis=1)

        horiz_dist = np.sqrt(x_l**2 + y_l**2)
        valid_h = horiz_dist > 1e-6
        vert_angles = np.zeros_like(distances)
        vert_angles[valid_h] = np.arctan2(z_l[valid_h], horiz_dist[valid_h])

        fov_mask = (
            (distances >= LIVOX_MIN_RANGE_M)
            & (distances <= LIVOX_MAX_RANGE_M)
            & (vert_angles >= half_v_min_rad)
            & (vert_angles <= half_v_max_rad)
        )
        visible_idx = np.where(fov_mask)[0]
        if len(visible_idx) == 0:
            continue

        sample_idx = ray_rng.choice(
            visible_idx,
            min(max_rays_per_pose, len(visible_idx)),
            replace=False,
        )

        pairs_from_this_pose = 0

        for p_idx in sample_idx:
            T = source_pts[p_idx]
            ray_vec = T - P
            ray_len = distances[p_idx]

            total_rays_checked += 1

            neigh_idx = ref_tree.query_ball_point(P, r=ray_len - tolerance)
            if not neigh_idx:
                continue

            neigh_pts = ref_pts[neigh_idx]
            ray_dir = ray_vec / ray_len
            vecs = neigh_pts - P
            proj = np.dot(vecs, ray_dir)
            valid_proj = (proj > 0.1) & (proj < ray_len - tolerance)
            if not np.any(valid_proj):
                continue

            vecs_v   = vecs[valid_proj]
            proj_v   = proj[valid_proj]
            idx_v    = np.array(neigh_idx)[valid_proj]
            candidate_pts = ref_pts[idx_v]

            perp = vecs_v - np.outer(proj_v, ray_dir)
            perp_dist = np.linalg.norm(perp, axis=1)

            collision = perp_dist < collision_radius
            if not np.any(collision):
                continue

            # --- кандидаты на нарушение (все, у которых луч проходит рядом) ---
            collided_idx   = idx_v[collision]
            collided_pts   = candidate_pts[collision]
            collided_perp  = perp_dist[collision]
            collided_proj  = proj_v[collision]        # ← расстояния от P вдоль луча

            # --- Фильтр 1: 3D-расстояние до целевой точки T ---
            d_to_T = np.linalg.norm(collided_pts - T, axis=1)
            same_surface_mask = d_to_T < surface_margin
            skipped_by_surface += int(np.sum(same_surface_mask))

            # --- Фильтр 2: KNN в source (есть ли рядом точка source) ---
            if surface_knn_check:
                knn_same_surface = np.zeros(len(collided_pts), dtype=bool)
                for k, X_b in enumerate(collided_pts):
                    if same_surface_mask[k]:
                        continue
                    dist_knn, _ = source_tree.query(X_b, k=1)
                    if dist_knn < surface_knn_radius:
                        knn_same_surface[k] = True
                skipped_by_surface_knn += int(np.sum(knn_same_surface))
                same_surface_mask |= knn_same_surface

            real_violation_mask = ~same_surface_mask
            if not np.any(real_violation_mask):
                continue

            # ============================================================
            # ГЛАВНОЕ ИЗМЕНЕНИЕ: берём ТОЛЬКО БЛИЖАЙШУЮ К P точку
            # (первую, в которую "упёрся" бы луч)
            # ============================================================
            real_idx    = collided_idx[real_violation_mask]
            real_pts    = collided_pts[real_violation_mask]
            real_perp   = collided_perp[real_violation_mask]
            real_proj   = collided_proj[real_violation_mask]   # ← distances from P

            # Индекс первой (= ближайшей) точки-нарушителя на луче
            first_local = int(np.argmin(real_proj))
            first_gi    = int(real_idx[first_local])
            first_pt    = real_pts[first_local]

            # Счётчики
            total_contradictions += 1
            skipped_not_first += int(len(real_idx) - 1)

            # Только ОДНА точка-нарушитель на этом луче
            bad_ref_idx_set.add(first_gi)

            if pairs_from_this_pose < max_pairs_per_pose:
                violation_pairs.append((P.copy(), first_pt))
                pairs_from_this_pose += 1

        if local_i % max(1, len(sampled_idx) // 10) == 0:
            print(f"    [{local_i}/{len(sampled_idx)}] лучей: {total_rays_checked}, "
                  f"нарушений: {total_contradictions}, "
                  f"пропущено как 'surface': {skipped_by_surface}, "
                  f"не-первых: {skipped_not_first}")

    if len(violation_pairs) > max_violation_pairs:
        sel = np.random.RandomState(ray_seed).choice(
            len(violation_pairs), max_violation_pairs, replace=False
        )
        violation_pairs = [violation_pairs[i] for i in sel]

    rate = (total_contradictions / total_rays_checked * 100) if total_rays_checked else 0.0
    score = 100.0 - rate

    print("\n" + "=" * 60)
    print("        РЕЗУЛЬТАТЫ ПРОВЕРКИ")
    print("=" * 60)
    print(f" Всего поз проверено:              {len(sampled_idx)}")
    print(f" Всего лучей проверено:            {total_rays_checked}")
    print(f" Нарушений (real):                 {total_contradictions}")
    print(f" Отфильтровано как 'та же плоскость' (к T):    {skipped_by_surface}")
    print(f" Отфильтровано по KNN source:                  {skipped_by_surface_knn}")
    print(f" Отброшено 'не первая на луче':                {skipped_not_first}")
    print(f" Процент нарушений:                {rate:.2f}%")
    print(f" Оценка согласованности:           {score:.2f}%")
    print(f" Уникальных плохих точек target:   {len(bad_ref_idx_set)}")
    print(f" Пар для визуализации:             {len(violation_pairs)}")
    print("=" * 60)

    return {
        "total_rays": total_rays_checked,
        "contradictions": total_contradictions,
        "contradiction_rate": rate,
        "consistency_score": score,
        "skipped_by_surface": skipped_by_surface,
        "skipped_by_surface_knn": skipped_by_surface_knn,
        "skipped_not_first": skipped_not_first,
        "bad_ref_idx": np.array(sorted(bad_ref_idx_set), dtype=np.int64),
        "violation_pairs": violation_pairs,
        "sampled_pose_idx": sampled_idx,
        "n_poses_total": n_poses_total,
        "n_poses_used": len(sampled_idx),
    }


def visualize_violations(
    cloud_source,
    cloud_reference,
    trajectory_positions,
    metrics,
    max_source_display=50000,
    max_ref_display=50000,
    max_bad_display=20000,
    max_lines=1500,
    pose_marker_radius=0.4,
):
    """
    Визуализация: source (серый), reference (синий), плохие точки (красный),
    траектория (зелёная), позы с нарушениями (жёлтые сферы),
    лучи от поз к плохим точкам (цветные).
    """
    print("\n[vis] Подготовка визуализации...")
    src_pts = np.asarray(cloud_source.points)
    ref_pts = np.asarray(cloud_reference.points)
    geoms = []

    # source
    src_vis = src_pts[np.random.RandomState(0).choice(len(src_pts), min(len(src_pts), max_source_display), replace=False)] if len(src_pts) > max_source_display else src_pts
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_vis)
    pcd_src.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(pcd_src)

    # reference
    ref_vis = ref_pts[np.random.RandomState(1).choice(len(ref_pts), min(len(ref_pts), max_ref_display), replace=False)] if len(ref_pts) > max_ref_display else ref_pts
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(ref_vis)
    pcd_ref.paint_uniform_color([0.2, 0.4, 1.0])
    geoms.append(pcd_ref)

    # плохие точки
    bad_idx = metrics.get("bad_ref_idx", np.array([], dtype=np.int64))
    if len(bad_idx) > 0:
        bad_vis = bad_idx[np.random.RandomState(2).choice(len(bad_idx), min(len(bad_idx), max_bad_display), replace=False)] if len(bad_idx) > max_bad_display else bad_idx
        pcd_bad = o3d.geometry.PointCloud()
        pcd_bad.points = o3d.utility.Vector3dVector(ref_pts[bad_vis])
        pcd_bad.paint_uniform_color([1.0, 0.1, 0.1])
        geoms.append(pcd_bad)
        print(f"[vis] плохих точек показано: {len(bad_vis)} из {len(bad_idx)}")

    # траектория
    if len(trajectory_positions) > 1:
        traj_ls = o3d.geometry.LineSet()
        traj_ls.points = o3d.utility.Vector3dVector(trajectory_positions)
        traj_ls.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(trajectory_positions)-1)])
        traj_ls.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(trajectory_positions)-1)])
        geoms.append(traj_ls)

    # позы, откуда есть нарушения
    pairs = metrics.get("violation_pairs", [])
    if pairs:
        unique_P = list({tuple(np.round(p, 3)) for p, _ in pairs})
        unique_P = np.array(unique_P)
        spheres = []
        import matplotlib.cm as cm
        palette = cm.get_cmap("hsv")(np.linspace(0, 1, len(unique_P)))[:, :3]
        for k, P in enumerate(unique_P):
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=pose_marker_radius)
            sph.translate(P)
            sph.paint_uniform_color(palette[k])
            spheres.append(sph)
        if spheres:
            merged = spheres[0]
            for s in spheres[1:]:
                merged += s
            geoms.append(merged)
            print(f"[vis] поз с нарушениями: {len(unique_P)}")

        # линии от поз к плохим точкам
        if len(pairs) > max_lines:
            sel = np.random.RandomState(3).choice(len(pairs), max_lines, replace=False)
            pairs_vis = [pairs[i] for i in sel]
        else:
            pairs_vis = pairs

        line_pts = []
        line_idx = []
        line_col = []
        P_to_color = {}
        for k, P in enumerate(unique_P):
            P_to_color[tuple(np.round(P, 3))] = palette[k]

        for (P, X) in pairs_vis:
            i = len(line_pts)
            line_pts.append(P)
            line_pts.append(X)
            line_idx.append([i, i+1])
            key = tuple(np.round(P, 3))
            line_col.append(P_to_color.get(key, [1.0, 0.5, 0.0]))

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(line_pts))
        ls.lines = o3d.utility.Vector2iVector(line_idx)
        ls.colors = o3d.utility.Vector3dVector(np.array(line_col))
        geoms.append(ls)
        print(f"[vis] линий показано: {len(pairs_vis)}")

    print("[vis] Легенда: серый=source, синий=target, красный=плохие точки target,")
    print("       зелёный=траектория, жёлтые сферы=позы с нарушениями, цветные линии=лучи")

    o3d.visualization.draw_geometries(geoms, window_name="Free-space violations", width=1400, height=800)
