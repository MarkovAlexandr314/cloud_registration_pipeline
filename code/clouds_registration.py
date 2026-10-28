# ============================================================
# Импорты
# ============================================================
import numpy as np
import open3d as o3d

from pathlib import Path
from scipy.spatial.transform import Rotation
from config_loader import load_config
from merger import iterative_merge

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
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml",
                    help="Путь к YAML-конфигу")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Переопределяем глобальный путь к TEASER из конфига
    TEASER_BIN = cfg["teaser_bin"]

    clouds, trajs, labels, groups = {}, {}, {}, {}
    for s in cfg["sessions"]:
        sid = s["id"]
        clouds[sid] = load_cloud(s["cloud"])
        traj_data = load_trajectory(s["traj"])
        if isinstance(traj_data, tuple):
            ts, pos, quats = traj_data
            traj_data = np.hstack([ts[:, None], pos, quats])
        trajs[sid]  = np.asarray(traj_data)
        labels[sid] = s["label"]
        groups[sid] = [sid]
        print(f"[load] id={sid}: {len(clouds[sid].points)} pts, "
              f"traj {trajs[sid].shape}")

    tcfg = cfg["teaser"]
    mcfg = cfg["merge"]

    final_clouds, final_trajs, final_labels, final_groups, history = iterative_merge(
        clouds, trajs, labels, groups, TEASER_BIN,
        merge_prob_threshold = mcfg["prob_threshold"],
        fsv_rate_hard_max    = mcfg["fsv_rate_hard_max"],
        fsv_max_poses        = mcfg["fsv_max_poses"],
        fsv_max_rays         = mcfg["fsv_max_rays"],
        max_iterations       = mcfg["max_iterations"],
        out_dir              = cfg["output_dir"],
        verbose              = True,
        vis_pair_ids         = None,
        vis_at_iter          = None,
        vis_traj_groups      = None,
        vis_traj_once        = False,
        teaser_voxel         = tcfg["voxel"],
        teaser_noise_bound   = tcfg["noise_bound"],
        ckpt_dir             = Path(cfg["checkpoint_dir"]),
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
