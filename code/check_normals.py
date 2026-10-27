"""Согласованность нормалей двух облаков + визуализация.

python check_normals.py --cloud-a a.pcd --cloud-b b.pcd
python check_normals.py --cloud-a a.pcd --cloud-b b.pcd --vis all
python check_normals.py --cloud-a a.pcd --cloud-b b.pcd --vis none
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def normals(pcd, knn=30, orient=True):
    # if not pcd.has_normals():
    ret = pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    if orient:
        try:
            pcd.orient_normals_consistent_tangent_plane(k=30)
        except Exception:
            pass
    return pcd


def common_points(pa, pb, tau=None, k_density=3.0, workers=-1):
    pts_a, pts_b = np.asarray(pa.points), np.asarray(pb.points)
    if tau is None:
        tau = k_density * float(np.mean(pa.compute_nearest_neighbor_distance()))
    d, idx_b = cKDTree(pts_b).query(pts_a, k=1, workers=workers)
    m = d < tau
    return np.where(m)[0], idx_b[m], tau


def normal_error_metric(pa, pb, tau=None, k_density=3.0,
                        angle_threshold_deg=30.0, knn=30,
                        orient=True, workers=-1, verbose=True, details=False):
    normals(pa, knn, orient)
    normals(pb, knn, orient)
    idx_a, idx_b, tau = common_points(pa, pb, tau, k_density, workers)
    n = len(idx_a)
    res = dict(n_common=n, tau=float(tau), mean_angle_deg=float("nan"),
               median_angle_deg=float("nan"), p95_angle_deg=float("nan"),
               max_angle_deg=float("nan"), consistency=0.0, normal_score=0.0)
    if n < 10:
        if verbose:
            print(f"    [normal] мало общих точек: {n}")
        return res

    na = np.asarray(pa.normals)[idx_a]
    nb = np.asarray(pb.normals)[idx_b]
    angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(na * nb, 1)), 0, 1)))

    mean = float(np.mean(angles))
    cons = float(np.mean(angles < angle_threshold_deg))
    score = 0.5 * max(0.0, 1.0 - mean / 90.0) + 0.5 * cons
    res.update(
        mean_angle_deg=mean,
        median_angle_deg=float(np.median(angles)),
        p95_angle_deg=float(np.percentile(angles, 95)),
        max_angle_deg=float(np.max(angles)),
        consistency=cons, normal_score=float(score),
    )
    if verbose:
        print(f"    [normal] n={n} mean={mean:.2f}° "
              f"med={res['median_angle_deg']:.2f}° cons={cons:.3f} "
              f"score={score:.4f}")
    if details:
        res.update(angles_deg=angles, idx_a=idx_a, na=na, nb=nb)
    return res


def _rgb(angles, cmap="viridis"):
    try:
        import matplotlib
        try:
            c = matplotlib.colormaps[cmap]
        except Exception:
            c = matplotlib.cm.get_cmap(cmap)
        return c(np.clip(angles / 90.0, 0, 1))[:, :3]
    except Exception:
        t = np.clip(angles / 90.0, 0, 1)
        return np.column_stack([t, 0.3 + 0.4 * t * (1 - t), 1 - t])


def _lines(points, ns, length, color):
    m = len(points)
    if m == 0:
        return None
    pts = np.empty((2 * m, 3))
    pts[0::2], pts[1::2] = points, points + length * ns
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(
        np.column_stack([np.arange(0, 2 * m, 2), np.arange(1, 2 * m, 2)])
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (m, 1)))
    return ls


def visualize(pa, res, mode="colored", cmap="viridis",
              max_lines=1500, length=None, window_name="Normal errors"):
    geoms = []
    if "angles_deg" not in res or len(res["angles_deg"]) == 0:
        print("    [vis] нет общих точек")
        return

    pts_a = np.asarray(pa.points)
    if mode in ("colored", "all"):
        colors = np.tile([0.55, 0.55, 0.55], (len(pts_a), 1))
        colors[res["idx_a"]] = _rgb(res["angles_deg"], cmap)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_a)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        geoms.append(pcd)

    if mode in ("normals", "all"):
        m = len(res["idx_a"])
        sel = (np.random.RandomState(0).choice(m, max_lines, replace=False)
               if m > max_lines else np.arange(m))
        pts = pts_a[res["idx_a"][sel]]
        if length is None:
            length = max(float(np.mean(pa.compute_nearest_neighbor_distance())) * 20, 0.1)
        for ls in (_lines(pts, res["na"][sel], length, [1, 0.2, 0.2]),
                   _lines(pts, res["nb"][sel], length, [0.2, 0.5, 1])):
            if ls is not None:
                geoms.append(ls)

    if geoms:
        o3d.visualization.draw_geometries(geoms, window_name=window_name)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cloud-a", required=True)
    p.add_argument("--cloud-b", required=True)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--k-density", type=float, default=3.0)
    p.add_argument("--angle-threshold", type=float, default=30.0)
    p.add_argument("--normals-knn", type=int, default=30)
    p.add_argument("--no-orient", action="store_true")
    p.add_argument("--workers", type=int, default=-1)
    p.add_argument("--vis", choices=["none", "colored", "normals", "all"],
                   default="colored")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--max-lines", type=int, default=1500)
    p.add_argument("--normal-length", type=float, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    pa = o3d.io.read_point_cloud(args.cloud_a)
    pa = pa.voxel_down_sample(voxel_size=0.05)
    pb = o3d.io.read_point_cloud(args.cloud_b)
    pb = pb.voxel_down_sample(voxel_size=0.05)
    print(f"[load] A: {len(pa.points)} точек")
    print(f"[load] B: {len(pb.points)} точек")

    res = normal_error_metric(
        pa, pb, tau=args.tau, k_density=args.k_density,
        angle_threshold_deg=args.angle_threshold, knn=args.normals_knn,
        orient=not args.no_orient, workers=args.workers,
        verbose=not args.quiet, details=(args.vis != "none"),
    )

    print("\n" + "=" * 56)
    print(f" Общих точек:         {res['n_common']}")
    print(f" tau:                 {res['tau']:.4f}")
    print(f" Средний угол:        {res['mean_angle_deg']:.2f}°")
    print(f" Медианный угол:      {res['median_angle_deg']:.2f}°")
    print(f" p95 угол:            {res['p95_angle_deg']:.2f}°")
    print(f" Макс. угол:          {res['max_angle_deg']:.2f}°")
    print(f" Consistency(<{args.angle_threshold:.0f}°): {res['consistency']:.3f}")
    print(f" normal_score:        {res['normal_score']:.4f}")
    print("=" * 56)

    if args.vis != "none":
        visualize(pa, res, mode=args.vis, cmap=args.cmap,
                  max_lines=args.max_lines, length=args.normal_length)


if __name__ == "__main__":
    main()

















# import numpy as np
# import open3d as o3d
# from scipy.spatial import cKDTree


# # ============================================================
# # 1. Утилита: оценка нормалей
# # ============================================================
# def estimate_normals_local(pcd, radius=None, orient=True):
#     """
#     Считает нормали у облака. Если нормали уже есть и radius=None — не трогает.
#     radius: если None — берётся 3 × avg_nn.
#     orient: согласовать ориентацию нормалей (orient_normals_consistent_tangent_plane).
#     """
#     if not pcd.has_normals() or radius is not None:
#         if radius is None:
#             avg_nn = float(np.mean(pcd.compute_nearest_neighbor_distance()))
#             radius = 3.0 * avg_nn
#         pcd.estimate_normals(
#             # o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
#             o3d.geometry.KDTreeSearchParamKNN(knn=8)
#         )
#     if orient:
#         try:
#             pcd.orient_normals_consistent_tangent_plane(k=30)
#         except Exception:
#             pass   # падает на очень разреженных облаках — пропускаем
#     return pcd


# # ============================================================
# # 2. Утилита: общие точки
# # ============================================================
# def find_common_points(pcd_a, pcd_b, tau=None, k_density=3.0):
#     """
#     Находит пары близких точек между облаками A и B.

#     tau: порог расстояния. Если None — берётся k_density × avg_nn(A).

#     Возвращает:
#         idx_a : индексы точек A
#         idx_b : соответствующие индексы точек B
#         tau   : фактический порог
#     """
#     pts_a = np.asarray(pcd_a.points)
#     pts_b = np.asarray(pcd_b.points)

#     if tau is None:
#         avg_nn = float(np.mean(pcd_a.compute_nearest_neighbor_distance()))
#         tau = k_density * avg_nn

#     tree_b = cKDTree(pts_b)
#     d_ab, idx_ab = tree_b.query(pts_a, k=1, workers=-1)

#     mask_a = d_ab < tau
#     idx_a = np.where(mask_a)[0]
#     idx_b = idx_ab[mask_a]

#     return idx_a, idx_b, tau


# # ============================================================
# # 3. ГЛАВНОЕ: метрика согласованности нормалей
# # ============================================================
# def normal_error_metric(pcd_a, pcd_b, tau=None, k_density=3.0,
#                         angle_threshold_deg=30.0, verbose=True):
#     """
#     Проверяет согласованность нормалей между двумя облаками.

#     Логика:
#       1. Для каждой точки A ищется ближайшая точка B.
#       2. Если расстояние < tau — пара считается «общей».
#       3. В общих точках считается угол между нормалями (через |cos| —
#          потому что нормаль определена с точностью до знака).
#       4. Собирается статистика: средний/медианный угол, consistency,
#          итоговый normal_score ∈ [0, 1].

#     Возвращает словарь:
#       n_common         — сколько общих точек
#       mean_angle_deg   — средний угол между нормалями
#       median_angle_deg — медианный угол
#       p95_angle_deg    — 95-й перцентиль
#       max_angle_deg    — максимальный угол
#       consistency      — доля пар с углом < angle_threshold_deg
#       normal_score     — итог в [0, 1] (больше = лучше)
#       tau              — использованный порог
#     """
#     # --- Нормали ---
#     estimate_normals_local(pcd_a)
#     estimate_normals_local(pcd_b)

#     n_a = np.asarray(pcd_a.normals)
#     n_b = np.asarray(pcd_b.normals)

#     # --- Общие точки ---
#     idx_a, idx_b, tau = find_common_points(pcd_a, pcd_b,
#                                             tau=tau, k_density=k_density)
#     n_common = len(idx_a)

#     if n_common < 10:
#         if verbose:
#             print(f"    [normal] мало общих точек: {n_common}")
#         return {
#             "n_common":          n_common,
#             "mean_angle_deg":    float("nan"),
#             "median_angle_deg":  float("nan"),
#             "p95_angle_deg":     float("nan"),
#             "max_angle_deg":     float("nan"),
#             "consistency":       0.0,
#             "normal_score":      0.0,
#             "tau":               float(tau),
#         }

#     # --- Углы между нормалями в общих точках ---
#     na = n_a[idx_a]
#     nb = n_b[idx_b]

#     # Берём модуль cos — нормали определены с точностью до знака
#     cos_angle = np.abs(np.sum(na * nb, axis=1))
#     cos_angle = np.clip(cos_angle, 0.0, 1.0)
#     angles_deg = np.degrees(np.arccos(cos_angle))
#     mean_angle   = float(np.mean(angles_deg))
#     median_angle = float(np.median(angles_deg))
#     p95_angle    = float(np.percentile(angles_deg, 95))
#     max_angle    = float(np.max(angles_deg))
#     consistency  = float(np.mean(angles_deg < angle_threshold_deg))

#     # --- Итоговый скор ---
#     angle_score = max(0.0, 1.0 - mean_angle / 90.0)
#     normal_score = 0.5 * angle_score + 0.5 * consistency

#     result = {
#         "n_common":          int(n_common),
#         "mean_angle_deg":    mean_angle,
#         "median_angle_deg":  median_angle,
#         "p95_angle_deg":     p95_angle,
#         "max_angle_deg":     max_angle,
#         "consistency":       consistency,
#         "normal_score":      float(normal_score),
#         "tau":               float(tau),
#     }

#     if verbose:
#         print(f"    [normal] n={n_common}  mean={mean_angle:.2f}°  "
#               f"med={median_angle:.2f}°  cons={consistency:.3f}  "
#               f"score={normal_score:.4f}")

#     return result


# # ============================================================
# # 4. MAIN — использование
# # ============================================================
# # if __name__ == "__main__":
# #     path_a = "/home/alex/Geoscan/code_cpp/geoscan_flight/rosbags_2/source_aligned.pcd"
# #     path_b = "/home/alex/Geoscan/code_cpp/geoscan_flight/rosbags_2/downsampled/cloud_5.pcd"

# #     # --- Загрузка ---
# #     pcd_a = o3d.io.read_point_cloud(path_a)
# #     pcd_b = o3d.io.read_point_cloud(path_b)
# #     print(f"Облако A: {len(pcd_a.points)} точек")
# #     print(f"Облако B: {len(pcd_b.points)} точек")

# #     # --- Проверка нормалей ---
# #     result = normal_error_metric(
# #         pcd_a, pcd_b,
# #         tau=None,                 # авто: 3 × avg_nn(A)
# #         k_density=3.0,
# #         angle_threshold_deg=30.0,
# #         verbose=True,
# #     )

# #     # --- Вывод результата ---
# #     print("\n" + "=" * 55)
# #     print("        СОГЛАСОВАННОСТЬ НОРМАЛЕЙ")
# #     print("=" * 55)
# #     print(f" Общих точек:              {result['n_common']}")
# #     print(f" Средний угол:             {result['mean_angle_deg']:.2f}°")
# #     print(f" Медианный угол:           {result['median_angle_deg']:.2f}°")
# #     print(f" 95-й перцентиль:          {result['p95_angle_deg']:.2f}°")
# #     print(f" Максимальный угол:        {result['max_angle_deg']:.2f}°")
# #     print(f" Consistency (< 30°):      {result['consistency']:.3f}")
# #     print(f" normal_score:             {result['normal_score']:.4f}")
# #     print("=" * 55)

# #     # --- Интерпретация ---
# #     ns = result['normal_score']
# #     if ns > 0.8:
# #         verdict = "ОТЛИЧНО — нормали хорошо согласованы"
# #     elif ns > 0.6:
# #         verdict = "ХОРОШО — облака согласованы"
# #     elif ns > 0.4:
# #         verdict = "СРЕДНЕ — серая зона, есть локальные проблемы"
# #     elif ns > 0.2:
# #         verdict = "ПЛОХО — согласованность слабая"
# #     else:
# #         verdict = "ОЧЕНЬ ПЛОХО — нормали не согласованы"

# #     print(f"\n→ {verdict}")