# ============================================================
# Импорты
# ============================================================
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import torch
import torch.nn as nn

from check_FSV import evaluate_free_space_consistency, visualize_violations
from check_normals import normal_error_metric
from NN import SimpleMLP


# ============================================================
# Конфиг
# ============================================================
TEASER_BIN = "/home/alex/Geoscan/libs/TEASER-plusplus/my_tools/build/teaser_register"
SAVE_DIR   = Path("checkpoints")

FEATURE_COLS = [
    "icp_fitness_fwd", "icp_fitness_bwd",
    "normal_score", "normal_mean_angle_deg",
    "fsv_rate_fwd", "fsv_rate_bwd",
]


# ============================================================
# Нейросетевой скоринг
# ============================================================

_MODEL_CACHE = {}


def load_final(ckpt_dir=SAVE_DIR):
    """Загружает финальный чекпоинт (с кэшем в памяти)."""
    ckpt_dir = Path(ckpt_dir).resolve()
    key = str(ckpt_dir)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    path = ckpt_dir / "final.pt"
    if not path.exists():
        raise FileNotFoundError(f"Финальный чекпоинт не найден: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SimpleMLP(input_dim=len(ckpt["feature_cols"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _MODEL_CACHE[key] = (model, ckpt)
    return model, ckpt


def predict_final(raw_rows, ckpt_dir=SAVE_DIR):
    """
    raw_rows: np.ndarray shape (n, len(FEATURE_COLS)) — в порядке FEATURE_COLS.
    Возвращает вероятности класса 1 (массив shape (n,)).
    """
    model, ckpt = load_final(ckpt_dir)
    expected = ckpt["feature_cols"]
    if list(expected) != list(FEATURE_COLS):
        raise RuntimeError(
            f"Порядок признаков не совпадает с чекпоинтом.\n"
            f"  checkpoint: {expected}\n  expected:  {FEATURE_COLS}")
    X = ckpt["imputer"].transform(raw_rows)
    X = ckpt["scaler"].transform(X)
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.FloatTensor(X))).numpy().flatten()
    return prob


def build_feature_row(icp_fitness_fwd, icp_fitness_bwd,
                       normal_score, normal_mean_angle_deg,
                       fsv_rate_fwd, fsv_rate_bwd):
    return np.array([[icp_fitness_fwd, icp_fitness_bwd,
                      normal_score, normal_mean_angle_deg,
                      fsv_rate_fwd, fsv_rate_bwd]], dtype=np.float64)


# ============================================================
# Утилиты загрузки
# ============================================================
VOXEL_SIZE = 0.15
def load_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        print(f"[!] Не удалось прочитать {path}")
        return None
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"{path}: {len(pcd.points)} точек")
    return pcd


def load_trajectory(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ts, pos, quats = data[:, 0], data[:, 1:4], data[:, 4:8]
    print(f"[traj] {path}: {len(ts)} поз")
    return ts, pos, quats


# ============================================================
# SE(3) helpers
# ============================================================
def is_valid_se3(T):
    if T is None:
        return False
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    if abs(np.linalg.det(R) - 1.0) > 1e-3:
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3):
        return False
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-6):
        return False
    return True


def transform_traj_batch(traj, T):
    ts, pos, quats = traj[:, 0], traj[:, 1:4], traj[:, 4:8]
    R, t = T[:3, :3], T[:3, 3]
    new_pos = (R @ pos.T).T + t
    rot_new = Rotation.from_matrix(R) * Rotation.from_quat(quats)
    return np.hstack([ts[:, None], new_pos, rot_new.as_quat()])


def T_from_traj_diff(new_traj, old_traj):
    old_pose = np.eye(4)
    old_pose[:3, :3] = Rotation.from_quat(old_traj[0, 4:8]).as_matrix()
    old_pose[:3, 3]  = old_traj[0, 1:4]
    new_pose = np.eye(4)
    new_pose[:3, :3] = Rotation.from_quat(new_traj[0, 4:8]).as_matrix()
    new_pose[:3, 3]  = new_traj[0, 1:4]
    return new_pose @ np.linalg.inv(old_pose)


# ============================================================
# TEASER++ через бинарник
# ============================================================
def teaser_registration(src_cloud, tgt_cloud, traj_tuple,
                        voxel=0.3, noise_bound=2.0,
                        verbose=False, timeout=600):
    if not os.path.exists(TEASER_BIN):
        raise FileNotFoundError(f"Нет бинарника TEASER++: {TEASER_BIN}")
    if not src_cloud.has_points() or not tgt_cloud.has_points():
        raise ValueError("Пустое облако на входе teaser_registration")

    if isinstance(traj_tuple, np.ndarray):
        traj = traj_tuple
    elif isinstance(traj_tuple, tuple) and len(traj_tuple) == 3:
        ts, pos, quats = traj_tuple
        traj = np.hstack([ts[:, None], pos, quats])
    else:
        raise ValueError(f"Неизвестный формат траектории: {type(traj_tuple)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "src.pcd")
        tgt_path = os.path.join(tmpdir, "tgt.pcd")
        out_path = os.path.join(tmpdir, "out.pcd")
        o3d.io.write_point_cloud(src_path, src_cloud)
        o3d.io.write_point_cloud(tgt_path, tgt_cloud)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        cmd = [TEASER_BIN, src_path, tgt_path, out_path,
               str(voxel), str(noise_bound)]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"teaser_register код {result.returncode}\n"
                f"STDERR: {result.stderr[-500:]}")

        m = re.search(
            r"\[final\][^\n]*:\s*\n"
            r"\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\n"
            r"\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\n"
            r"\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\n"
            r"\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)",
            result.stdout)
        if m is None:
            raise RuntimeError(
                f"Не удалось распарсить [final] T\n"
                f"STDOUT tail:\n{result.stdout[-1000:]}")

        vals = [float(m.group(i)) for i in range(1, 17)]
        T = np.array(vals).reshape(4, 4)

    if not is_valid_se3(T):
        raise RuntimeError(f"teaser_register вернул невалидный SE(3):\n{T}")

    src_aligned_full = o3d.geometry.PointCloud(src_cloud).transform(T)
    new_traj = transform_traj_batch(traj, T)
    return tgt_cloud, src_aligned_full, new_traj, T


# ============================================================
# Адаптивные параметры FSV
# ============================================================
def adaptive_fsv_params(n_merges, base_tol=0.2, base_margin=0.4):
    scale = 1.0 + 0.5 * n_merges
    return {
        "tolerance":          base_tol * scale,
        "collision_radius":   base_tol * scale,
        "surface_margin":     base_margin * scale,
        "surface_knn_radius": base_margin * scale,
        "scale":              scale,
    }


# ============================================================
# Итеративное слияние (без PGO, решение по нейросети)
# ============================================================
def iterative_merge(clouds, trajs, labels, groups,
                    merge_prob_threshold=0.5,
                    fsv_rate_hard_max=0.95,
                    fsv_max_poses=100,
                    fsv_max_rays=20,
                    max_iterations=15,
                    out_dir=".",
                    verbose=True,
                    vis_pair_ids=None,
                    vis_at_iter=None,
                    vis_pose_marker_radius=0.4,
                    vis_once=True,
                    vis_traj_groups=None,
                    vis_traj_once=True,
                    teaser_voxel=0.3,
                    teaser_noise_bound=2.0,
                    ckpt_dir=SAVE_DIR,
                    ):
    history = []
    iteration = 0
    pair_cache = {}
    versions = {cid: 0 for cid in clouds}

    cloud_sessions = {cid: [cid] for cid in clouds}

    vis_done = False
    vis_traj_done = False

    # Проверим, что сеть загружается и порядок признаков совпадает
    load_final(ckpt_dir)

    if verbose:
        print(f"[NN] ckpt={Path(ckpt_dir) / 'final.pt'}")
        print(f"[TEASER] voxel={teaser_voxel}, noise_bound={teaser_noise_bound}")

    while len(clouds) > 1 and iteration < max_iterations:
        iteration += 1
        if verbose:
            print(f"\n{'='*72}")
            print(f"ИТЕРАЦИЯ {iteration}  |  активных облаков: {len(clouds)}")
            print(f"{'='*72}")

        active = sorted(clouds.keys())
        pair_results = {}
        n_cached = n_computed = 0

        for ai in range(len(active)):
            for bi in range(ai + 1, len(active)):
                i, j = active[ai], active[bi]
                key = (i, j)

                cached = pair_cache.get(key)
                if (cached is not None
                        and cached["v_i"] == versions[i]
                        and cached["v_j"] == versions[j]):
                    n_cached += 1
                    if cached["result"] is None:
                        if verbose:
                            print(f"\n  Пара ({i},{j}): [CACHED — rejected]")
                        continue
                    pair_results[key] = cached["result"]
                    if verbose:
                        print(f"\n  Пара ({i},{j}): [CACHED]")
                    continue

                if verbose:
                    print(f"\n  Пара ({i},{j}): {labels[i]}  ↔  {labels[j]}")

                n_merges_max = max(len(cloud_sessions[i]) - 1,
                                    len(cloud_sessions[j]) - 1)
                fsv_params = adaptive_fsv_params(n_merges_max)

                # ----- TEASER -----
                try:
                    tgt_full, src_aligned_full, new_traj, T = teaser_registration(
                        clouds[i], clouds[j], trajs[i],
                        voxel=teaser_voxel,
                        noise_bound=teaser_noise_bound,
                        verbose=verbose,
                    )
                except Exception as e:
                    if verbose:
                        print(f"    [FAILED teaser] {e}")
                    pair_cache[key] = {"v_i": versions[i],
                                       "v_j": versions[j], "result": None}
                    continue

                T_inv = np.linalg.inv(T)

                # ----- ICP fitness fwd/bwd -----
                try:
                    ev_fwd = o3d.pipelines.registration.evaluate_registration(
                        src_aligned_full, tgt_full, 0.5, np.eye(4))
                    ev_bwd = o3d.pipelines.registration.evaluate_registration(
                        tgt_full, src_aligned_full, 0.5, np.eye(4))
                    icp_fitness_fwd = float(ev_fwd.fitness)
                    icp_fitness_bwd = float(ev_bwd.fitness)
                except Exception as e:
                    if verbose:
                        print(f"    [FAILED icp] {e}")
                    pair_cache[key] = {"v_i": versions[i],
                                       "v_j": versions[j], "result": None}
                    continue

                # ----- FSV forward: traj_i в фрейме j против tgt -----
                positions_f = new_traj[:, 1:4]
                orients_f   = new_traj[:, 4:8]
                try:
                    fsv_fwd = evaluate_free_space_consistency(
                        cloud_source=src_aligned_full,
                        cloud_reference=tgt_full,
                        trajectory_positions=positions_f,
                        trajectory_orientations=orients_f,
                        tolerance=fsv_params["tolerance"],
                        collision_radius=fsv_params["collision_radius"],
                        surface_margin=fsv_params["surface_margin"],
                        surface_knn_radius=fsv_params["surface_knn_radius"],
                        max_rays_per_pose=fsv_max_rays,
                        max_poses=fsv_max_poses,
                        pose_seed=42, ray_seed=42,
                        max_pairs_per_pose=30,
                        max_violation_pairs=5000,
                    )
                except Exception as e:
                    if verbose:
                        print(f"    [FAILED FSV-fwd] {e}")
                    pair_cache[key] = {"v_i": versions[i],
                                       "v_j": versions[j], "result": None}
                    continue
                fsv_rate_fwd = fsv_fwd["contradiction_rate"] / 100.0

                # Ранний отсев, чтобы не считать FSV-bwd зря
                if fsv_rate_fwd > fsv_rate_hard_max:
                    if verbose:
                        print(f"    [fsv_fwd={fsv_rate_fwd:.3f} > hard_max] reject")
                    pair_cache[key] = {"v_i": versions[i],
                                       "v_j": versions[j], "result": None}
                    continue

                # ----- FSV backward: traj_j в фрейме i против i -----
                try:
                    traj_j_aligned = transform_traj_batch(trajs[j], T_inv)
                    tgt_in_src_frame = o3d.geometry.PointCloud(
                        tgt_full).transform(T_inv)
                    positions_b = traj_j_aligned[:, 1:4]
                    orients_b   = traj_j_aligned[:, 4:8]
                    fsv_bwd = evaluate_free_space_consistency(
                        cloud_source=tgt_in_src_frame,
                        cloud_reference=clouds[i],
                        trajectory_positions=positions_b,
                        trajectory_orientations=orients_b,
                        tolerance=fsv_params["tolerance"],
                        collision_radius=fsv_params["collision_radius"],
                        surface_margin=fsv_params["surface_margin"],
                        surface_knn_radius=fsv_params["surface_knn_radius"],
                        max_rays_per_pose=fsv_max_rays,
                        max_poses=fsv_max_poses,
                        pose_seed=42, ray_seed=42,
                        max_pairs_per_pose=30,
                        max_violation_pairs=5000,
                    )
                except Exception as e:
                    if verbose:
                        print(f"    [FAILED FSV-bwd] {e}")
                    pair_cache[key] = {"v_i": versions[i],
                                       "v_j": versions[j], "result": None}
                    continue
                fsv_rate_bwd = fsv_bwd["contradiction_rate"] / 100.0

                if fsv_rate_bwd > fsv_rate_hard_max:
                    if verbose:
                        print(f"    [fsv_bwd={fsv_rate_bwd:.3f} > hard_max] reject")
                    pair_cache[key] = {"v_i": versions[i],
                                       "v_j": versions[j], "result": None}
                    continue

                # ----- Нормали -----
                ne = normal_error_metric(tgt_full, src_aligned_full,
                                         verbose=verbose)
                normal_score = float(ne["normal_score"])
                normal_mean_angle_deg = float(
                    ne.get("normal_mean_angle_deg",
                           ne.get("mean_angle_deg",
                                  ne.get("normal_angle_mean_deg", 0.0))))

                # ----- Нейросетевой скоринг -----
                feats = build_feature_row(
                    icp_fitness_fwd, icp_fitness_bwd,
                    normal_score, normal_mean_angle_deg,
                    fsv_rate_fwd, fsv_rate_bwd)
                try:
                    prob = float(predict_final(feats, ckpt_dir=ckpt_dir)[0])
                except Exception as e:
                    if verbose:
                        print(f"    [FAILED NN] {e}")
                    pair_cache[key] = {"v_i": versions[i],
                                       "v_j": versions[j], "result": None}
                    continue

                if verbose:
                    print(f"    feats: icp=({icp_fitness_fwd:.3f}/{icp_fitness_bwd:.3f}) "
                          f"normal={normal_score:.3f}/{normal_mean_angle_deg:.2f}deg "
                          f"fsv=({fsv_rate_fwd:.3f}/{fsv_rate_bwd:.3f}) "
                          f"→ P={prob:.3f}")

                if prob < merge_prob_threshold:
                    if verbose:
                        print(f"    [NN] {prob:.3f} < {merge_prob_threshold} — reject")
                    pair_cache[key] = {"v_i": versions[i],
                                       "v_j": versions[j], "result": None}
                    continue

                # ----- Визуализация FSV -----
                if vis_traj_groups is not None and not vis_traj_done:
                    g_target_i = sorted(vis_traj_groups[0])
                    g_target_j = sorted(vis_traj_groups[1])
                    g_cur_i = sorted(groups[i])
                    g_cur_j = sorted(groups[j])
                    if ((g_cur_i == g_target_i and g_cur_j == g_target_j) or
                        (g_cur_i == g_target_j and g_cur_j == g_target_i)):
                        print(f"\n{'#'*72}")
                        print(f"# VIS-TRAJ pair=({i},{j}) iter={iteration} "
                              f"P={prob:.3f} fsv_fwd={fsv_rate_fwd:.3f}")
                        print(f"{'#'*72}")
                        visualize_violations(
                            cloud_source=src_aligned_full,
                            cloud_reference=tgt_full,
                            trajectory_positions=positions_f,
                            metrics=fsv_fwd,
                            pose_marker_radius=vis_pose_marker_radius,
                        )
                        if vis_traj_once:
                            vis_traj_done = True

                if (vis_pair_ids is not None
                        and (i, j) == tuple(vis_pair_ids)
                        and (vis_at_iter is None or iteration == vis_at_iter)
                        and (not vis_done or not vis_once)):
                    print(f"\n# VIS-FSV pair=({i},{j}) iter={iteration}")
                    visualize_violations(
                        cloud_source=src_aligned_full,
                        cloud_reference=tgt_full,
                        trajectory_positions=positions_f,
                        metrics=fsv_fwd,
                        pose_marker_radius=vis_pose_marker_radius,
                    )
                    vis_done = True

                result = {
                    "merged":              tgt_full + src_aligned_full,
                    "aligned_src":         src_aligned_full,
                    "new_traj":            new_traj,
                    "T":                   T,
                    "fsv_fwd":             fsv_fwd,
                    "fsv_bwd":             fsv_bwd,
                    "fsv_rate_fwd":        fsv_rate_fwd,
                    "fsv_rate_bwd":        fsv_rate_bwd,
                    "normal_metric":       ne,
                    "normal_score":        normal_score,
                    "normal_mean_angle_deg": normal_mean_angle_deg,
                    "icp_fitness_fwd":     icp_fitness_fwd,
                    "icp_fitness_bwd":     icp_fitness_bwd,
                    "merge_probability":   prob,
                }
                pair_results[key] = result
                pair_cache[key] = {
                    "v_i": versions[i], "v_j": versions[j], "result": result}
                n_computed += 1

        if verbose:
            print(f"\n  [кэш] переиспользовано: {n_cached}, "
                  f"пересчитано: {n_computed}")

        if not pair_results:
            if verbose:
                print("\n  Нет разрешённых пар — стоп.")
            break

        # Лучшая пара — по вероятности сети
        best_pair = max(pair_results,
                        key=lambda k: pair_results[k]["merge_probability"])
        best = pair_results[best_pair]
        i, j = best_pair

        if verbose:
            print(f"\n  >>> ЛУЧШАЯ ПАРА: ({i},{j})  "
                  f"P={best['merge_probability']:.4f}")

        if best["merge_probability"] < merge_prob_threshold:
            if verbose:
                print("  >>> ниже порога — стоп")
            break

        # ----- Слияние (без PGO) -----
        clouds[i] = best["merged"]                              # в фрейме j
        trajs[i]  = np.vstack([best["new_traj"], trajs[j]])     # обе в фрейме j
        cloud_sessions[i] = cloud_sessions[i] + cloud_sessions[j]
        groups[i] = groups[i] + groups[j]
        labels[i] = f"[{labels[i]} + {labels[j]}]"
        versions[i] += 1

        del clouds[j]; del trajs[j]; del groups[j]; del labels[j]
        del versions[j]; del cloud_sessions[j]

        for k in list(pair_cache.keys()):
            if i in k or j in k:
                del pair_cache[k]

        merged_path = os.path.join(out_dir,
                                    f"merged_iter{iteration}_id{i}.pcd")
        o3d.io.write_point_cloud(merged_path, clouds[i])

        history.append({
            "iter":                 iteration,
            "pair":                 best_pair,
            "merge_probability":    best["merge_probability"],
            "icp_fitness_fwd":      best["icp_fitness_fwd"],
            "icp_fitness_bwd":      best["icp_fitness_bwd"],
            "normal_score":         best["normal_score"],
            "normal_mean_angle_deg": best["normal_mean_angle_deg"],
            "fsv_rate_fwd":         best["fsv_rate_fwd"],
            "fsv_rate_bwd":         best["fsv_rate_bwd"],
            "groups":               list(groups[i]),
            "path":                 merged_path,
        })

        if verbose:
            print(f"  >>> Слиты {i} + {j} → новое облако {i}")
            print(f"  >>> Активных облаков: {len(clouds)}")

    return clouds, trajs, labels, groups, history


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    
    # ============================================================
    # DATASET FROM HANGAR
    ROOT = "/home/alex/Geoscan/code_cpp/geoscan_flight/hangar/ros_2_3/"

    ids = [11, 13, 14, 15, 16, 20, 21, 22, 23, 24]
    cloud_paths = [ROOT + f"downsampled/cloud_{i}.pcd" for i in ids]
    traj_paths  = [ROOT + f"paths/path_{i}.txt"        for i in ids]
    
    # ============================================================
    # DATASET FROM CAVE
    # ROOT = "/home/alex/Geoscan/code_cpp/geoscan_flight/cave/"

    # ids = [i for i in range(8)]
    # cloud_paths = [ROOT + f"raw/bag_{i}.pcd" for i in ids]
    # traj_paths  = [ROOT + f"paths/path_{i}.txt" for i in ids]
    # ============================================================


    clouds, trajs, labels, groups = {}, {}, {}, {}
    for idx, (cp, tp) in enumerate(zip(cloud_paths, traj_paths)):
        if not os.path.exists(cp) or not os.path.exists(tp):
            print(f"[skip] отсутствует {cp} или {tp}")
            continue
        clouds[idx] = load_cloud(cp)
        traj_data = load_trajectory(tp)
        if isinstance(traj_data, tuple):
            ts, pos, quats = traj_data
            traj_data = np.hstack([ts[:, None], pos, quats])
        trajs[idx]  = np.asarray(traj_data)
        labels[idx] = os.path.basename(cp)
        groups[idx] = [idx]
        print(f"[load] cloud {idx}: {len(clouds[idx].points)} pts, "
              f"traj {trajs[idx].shape}")

    final_clouds, final_trajs, final_labels, final_groups, history = iterative_merge(
        clouds, trajs, labels, groups,
        merge_prob_threshold=0.5,
        fsv_rate_hard_max=0.95,
        fsv_max_poses=100,
        fsv_max_rays=30,
        max_iterations=15,
        out_dir=".",
        verbose=True,
        vis_pair_ids=None,
        vis_at_iter=None,
        vis_traj_groups=None,
        vis_traj_once=False,
        teaser_voxel=0.3,
        teaser_noise_bound=0.5,
        ckpt_dir=Path("checkpoints"),
    )

    print("\n" + "=" * 72)
    print("ИТОГ")
    print("=" * 72)
    for h in history:
        print(f"  iter {h['iter']:>2}  pair={h['pair']}  "
              f"P={h['merge_probability']:.4f}  "
              f"icp=({h['icp_fitness_fwd']:.3f}/{h['icp_fitness_bwd']:.3f})  "
              f"normal=({h['normal_score']:.3f}/{h['normal_mean_angle_deg']:.1f}°)  "
              f"fsv=({h['fsv_rate_fwd']:.3f}/{h['fsv_rate_bwd']:.3f})")

    print(f"\nОсталось облаков: {len(final_clouds)}")
    for cid in sorted(final_clouds.keys()):
        print(f"  Облако {cid}: {final_labels[cid]}")
        print(f"    группы: {final_groups[cid]}")
        print(f"    точек:  {len(final_clouds[cid].points)}")

    if final_clouds:
        vis_list = []
        for cid in sorted(final_clouds.keys()):
            pcd = o3d.geometry.PointCloud(final_clouds[cid])
            pcd.paint_uniform_color(np.random.RandomState(cid).rand(3))
            vis_list.append(pcd)
        o3d.visualization.draw_geometries(vis_list,
                                            window_name="Финальные облака")