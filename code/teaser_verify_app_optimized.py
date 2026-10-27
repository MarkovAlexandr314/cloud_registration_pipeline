#!/usr/bin/env python3
"""
TEASER Verification Tool — мини-приложение для ручной верификации склейки
двух облаков. Пишет метрики + вердикт человека в JSONL (для обучения).
"""

import os
import re
import json
import time
import gc
import threading
import subprocess
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import check_normals

# ============================================================
# ОПТИМИЗАЦИЯ ПАМЯТИ: Отключаем orient_normals_consistent_tangent_plane
# ============================================================
def fast_estimate_normals_local(pcd, radius=None, orient=False):
    if not pcd.has_normals() or radius is not None:
        if radius is None:
            avg_nn = float(np.mean(pcd.compute_nearest_neighbor_distance()))
            radius = 3.0 * avg_nn
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
    return pcd

# Подменяем функцию в модуле check_normals на нашу (без orient)
check_normals.estimate_normals_local = fast_estimate_normals_local
from check_normals import normal_error_metric

from check_FSV_2 import evaluate_free_space_consistency


TEASER_BIN = "/home/alex/Geoscan/libs/TEASER-plusplus/my_tools/build/teaser_register"
VOXEL_SIZE = 0.15   # Увеличен для экономии памяти (было 0.2 + random 0.15)
PROC = 1      # УБРАНО: random_down_sample ломает структуру для TEASER


# ============================================================
# Core helpers
# ============================================================
def load_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"Пустое облако: {path}")
    return pcd


def load_trajectory_array(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        raise RuntimeError(f"Ожидалось 8 колонок в {path}, получено {data.shape[1]}")
    return np.asarray(data, dtype=np.float64)


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


def teaser_registration(src_cloud, tgt_cloud, traj,
                        voxel=0.3, noise_bound=2.0, timeout=600):
    if not os.path.exists(TEASER_BIN):
        raise FileNotFoundError(f"Нет TEASER++: {TEASER_BIN}")

    # Сохраняем оригинальные цвета (если были)
    src_had_colors = src_cloud.has_colors()
    src_colors = src_cloud.colors
    if not src_had_colors:
        src_cloud.paint_uniform_color([0.5, 0.5, 0.5])

    tgt_had_colors = tgt_cloud.has_colors()
    tgt_colors = tgt_cloud.colors
    if not tgt_had_colors:
        tgt_cloud.paint_uniform_color([0.5, 0.5, 0.5])

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "src.pcd")
        tgt_path = os.path.join(tmpdir, "tgt.pcd")
        out_path = os.path.join(tmpdir, "out.pcd")
        
        o3d.io.write_point_cloud(src_path, src_cloud)
        o3d.io.write_point_cloud(tgt_path, tgt_cloud)

        # Восстанавливаем/удаляем цвета, чтобы освободить память
        if not src_had_colors:
            src_cloud.colors = o3d.utility.Vector3dVector()
        else:
            src_cloud.colors = src_colors

        if not tgt_had_colors:
            tgt_cloud.colors = o3d.utility.Vector3dVector()
        else:
            tgt_cloud.colors = tgt_colors
            
        gc.collect()

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        cmd = [TEASER_BIN, src_path, tgt_path, out_path,
               str(voxel), str(noise_bound)]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, env=env)
        if res.returncode != 0:
            raise RuntimeError(
                f"teaser код {res.returncode}\nSTDERR: {res.stderr[-500:]}")

        m = re.search(
            r"\[final\][^\n]*:\s*\n"
            r"\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\n"
            r"\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\n"
            r"\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\n"
            r"\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)",
            res.stdout)
        if m is None:
            raise RuntimeError(f"Не удалось распарсить T:\n{res.stdout[-800:]}")
        T = np.array([float(m.group(i)) for i in range(1, 17)]).reshape(4, 4)

    if not is_valid_se3(T):
        raise RuntimeError(f"Невалидный SE(3):\n{T}")

    # Возвращаем копию, сдвинутую матрицей T
    src_aligned = o3d.geometry.PointCloud(src_cloud).transform(T)
    new_traj = transform_traj_batch(traj, T) if traj is not None else None
    return T, src_aligned, new_traj


# ============================================================
# GUI
# ============================================================
class TeaserVerifyApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"TEASER Verification Tool  (voxel={VOXEL_SIZE})")
        self.root.geometry("960x820")

        self.cloud_a_path = tk.StringVar()
        self.cloud_b_path = tk.StringVar()
        self.traj_a_path  = tk.StringVar()
        self.traj_b_path  = tk.StringVar()
        self.output_path  = tk.StringVar(value=os.path.abspath("teaser_verdicts.jsonl"))
        self.status_var   = tk.StringVar(value="Готов")
        self.save_status  = tk.StringVar(value="")

        self.cloud_a = self.cloud_b = None
        self.n_a_raw = self.n_b_raw = 0
        self.traj_a = self.traj_b = None
        self.T = None
        self.src_aligned = None
        self.new_traj_a = None
        self.metrics = {}
        self.computing = False
        self._loaded_key = None

        self._build_ui()

    def _build_ui(self):
        frm_in = ttk.LabelFrame(self.root,
                                text=f"1. Входные данные  (voxel={VOXEL_SIZE} м)")
        frm_in.pack(fill="x", padx=10, pady=5)
        self._file_row(frm_in, 0, "Cloud A (source)", self.cloud_a_path, "cloud")
        self._file_row(frm_in, 1, "Cloud B (target)", self.cloud_b_path, "cloud")
        self._file_row(frm_in, 2, "Traj A (опц.)",    self.traj_a_path,  "traj")
        self._file_row(frm_in, 3, "Traj B (опц.)",    self.traj_b_path,  "traj")

        frm_act = ttk.Frame(self.root)
        frm_act.pack(fill="x", padx=10, pady=5)
        ttk.Button(frm_act, text="Показать облака",
                   command=self.show_clouds).pack(side="left", padx=5)
        ttk.Button(frm_act, text="Запустить TEASER + метрики",
                   command=self.run_pipeline).pack(side="left", padx=5)
        ttk.Button(frm_act, text="Показать результат склейки",
                   command=self.show_merged).pack(side="left", padx=5)

        ttk.Label(self.root, textvariable=self.status_var,
                  foreground="blue").pack(fill="x", padx=10, pady=2)

        frm_m = ttk.LabelFrame(self.root, text="2. Метрики")
        frm_m.pack(fill="both", expand=True, padx=10, pady=5)
        self.metrics_text = tk.Text(frm_m, height=20, font=("Courier", 10), wrap="none")
        self.metrics_text.pack(fill="both", expand=True, padx=5, pady=5)

        frm_v = ttk.LabelFrame(self.root, text="3. Вердикт")
        frm_v.pack(fill="x", padx=10, pady=5)
        btn_row = ttk.Frame(frm_v)
        btn_row.pack(fill="x", padx=5, pady=5)
        self.btn_correct = ttk.Button(
            btn_row, text="✓ Правильная склейка",
            command=lambda: self.save_verdict("correct"), state="disabled")
        self.btn_correct.pack(side="left", padx=5)
        self.btn_incorrect = ttk.Button(
            btn_row, text="✗ Неправильная склейка",
            command=lambda: self.save_verdict("incorrect"), state="disabled")
        self.btn_incorrect.pack(side="left", padx=5)
        ttk.Button(btn_row, text="Сброс", command=self.reset_all).pack(side="left", padx=20)

        frm_out = ttk.Frame(frm_v)
        frm_out.pack(fill="x", padx=5, pady=5)
        ttk.Label(frm_out, text="Файл:").pack(side="left")
        ttk.Entry(frm_out, textvariable=self.output_path, width=60).pack(side="left", padx=5)
        ttk.Button(frm_out, text="...", command=self._pick_output).pack(side="left")

        ttk.Label(frm_v, textvariable=self.save_status,
                  foreground="green").pack(fill="x", padx=5, pady=2)

    def _file_row(self, parent, row, label, var, kind):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(parent, textvariable=var, width=72).grid(row=row, column=1, padx=5)
        ttk.Button(parent, text="Browse...",
                   command=lambda: self._pick_file(var, kind)).grid(row=row, column=2, padx=5)

    def _pick_file(self, var, kind):
        if kind == "cloud":
            path = filedialog.askopenfilename(
                filetypes=[("Point cloud", "*.pcd *.ply"), ("All", "*.*")])
        else:
            path = filedialog.askopenfilename(
                filetypes=[("Trajectory", "*.txt"), ("All", "*.*")])
        if path:
            var.set(path)

    def _pick_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL", "*.jsonl"), ("CSV", "*.csv")])
        if path:
            self.output_path.set(path)

    def _load_all_if_changed(self):
        key = (self.cloud_a_path.get(), self.cloud_b_path.get(),
               self.traj_a_path.get(), self.traj_b_path.get())
        if self._loaded_key == key and self.cloud_a is not None:
            return
        self._load_all()
        self._loaded_key = key

    def _load_all(self):
        cp_a = self.cloud_a_path.get().strip()
        cp_b = self.cloud_b_path.get().strip()
        tp_a = self.traj_a_path.get().strip()
        tp_b = self.traj_b_path.get().strip()

        if not cp_a or not cp_b:
            raise RuntimeError("Укажите пути к Cloud A и Cloud B")
        for p in (cp_a, cp_b):
            if not os.path.exists(p):
                raise RuntimeError(f"Файл не найден: {p}")

        raw_a = load_cloud(cp_a)
        raw_b = load_cloud(cp_b)
        self.n_a_raw = len(raw_a.points)
        self.n_b_raw = len(raw_b.points)

        print(f"[voxel] Cloud A: {self.n_a_raw} → ", end="", flush=True)
        self.cloud_a = raw_a.voxel_down_sample(voxel_size=VOXEL_SIZE).random_down_sample(sampling_ratio=PROC)
        # self.cloud_a = raw_a.farthest_point_down_sample(num_samples=40_000)
        del raw_a
        print(f"{len(self.cloud_a.points)} pts")

        print(f"[voxel] Cloud B: {self.n_b_raw} → ", end="", flush=True)
        self.cloud_b = raw_b.voxel_down_sample(voxel_size=VOXEL_SIZE).random_down_sample(sampling_ratio=PROC) # .random_down_sample(sampling_ratio=PROC)
        # self.cloud_b = raw_b.farthest_point_down_sample(num_samples=40_000)
        del raw_b
        # Вместо voxel_down_sample
        print(f"{len(self.cloud_b.points)} pts")
        
        gc.collect()

        if len(self.cloud_a.points) == 0 or len(self.cloud_b.points) == 0:
            raise RuntimeError(
                f"После даунсэмплинга (voxel={VOXEL_SIZE}) одно из облаков пустое.")

        self.traj_a = self.traj_b = None
        if tp_a:
            if not os.path.exists(tp_a):
                raise RuntimeError(f"Файл траектории не найден: {tp_a}")
            self.traj_a = load_trajectory_array(tp_a)
        if tp_b:
            if not os.path.exists(tp_b):
                raise RuntimeError(f"Файл траектории не найден: {tp_b}")
            self.traj_b = load_trajectory_array(tp_b)
        if (self.traj_a is None) != (self.traj_b is None):
            raise RuntimeError("Укажите обе траектории или ни одной")

    def _traj_lineset(self, traj, color):
        pos = traj[:, 1:4]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pos)
        ls.lines = o3d.utility.Vector2iVector(
            [[k, k + 1] for k in range(len(pos) - 1)])
        ls.colors = o3d.utility.Vector3dVector(
            [color for _ in range(len(pos) - 1)])
        return ls

    def show_clouds(self):
        try:
            self._load_all_if_changed()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return

        a = o3d.geometry.PointCloud(self.cloud_a)
        a.paint_uniform_color([1.0, 0.3, 0.3])
        b = o3d.geometry.PointCloud(self.cloud_b)
        b.paint_uniform_color([0.3, 0.5, 1.0])

        geoms = [a, b]
        if self.traj_a is not None:
            geoms.append(self._traj_lineset(self.traj_a, (1.0, 0.6, 0.6)))
            geoms.append(self._traj_lineset(self.traj_b, (0.6, 0.6, 1.0)))

        self.status_var.set(f"Облака: A={len(self.cloud_a.points)} pts, B={len(self.cloud_b.points)} pts")
        o3d.visualization.draw_geometries(geoms, window_name="A (red) + B (blue)", width=1400, height=800)

    def show_merged(self):
        if self.src_aligned is None:
            messagebox.showinfo("Нет данных", "Сначала запустите TEASER + метрики")
            return

        tgt = o3d.geometry.PointCloud(self.cloud_b)
        tgt.paint_uniform_color([0.3, 0.5, 1.0])
        src = o3d.geometry.PointCloud(self.src_aligned)
        src.paint_uniform_color([1.0, 0.3, 0.3])

        geoms = [tgt, src]
        if self.new_traj_a is not None and self.traj_b is not None:
            geoms.append(self._traj_lineset(self.new_traj_a, (1.0, 0.6, 0.6)))
            geoms.append(self._traj_lineset(self.traj_b, (0.6, 0.6, 1.0)))

        self.status_var.set("Результат склейки: A_aligned (red) + B (blue)")
        o3d.visualization.draw_geometries(geoms, window_name="Merged: A_aligned + B", width=1400, height=800)

    def run_pipeline(self):
        if self.computing:
            return
        self.computing = True
        self.btn_correct.config(state="disabled")
        self.btn_incorrect.config(state="disabled")
        self.metrics_text.delete("1.0", "end")
        self.status_var.set("Загрузка...")
        threading.Thread(target=self._pipeline_worker, daemon=True).start()

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _pipeline_worker(self):
        try:
            self._set_status(f"Загрузка + voxel={VOXEL_SIZE}...")
            self._load_all_if_changed()

            self._set_status("TEASER++...")
            t0 = time.time()
            self.T, self.src_aligned, self.new_traj_a = teaser_registration(
                self.cloud_a, self.cloud_b, self.traj_a, voxel=0.3, noise_bound=2.0)
            t_teaser = time.time() - t0
            gc.collect()

            self._set_status("ICP fitness...")
            ev_f = o3d.pipelines.registration.evaluate_registration(
                self.src_aligned, self.cloud_b, 0.5, np.eye(4))
            ev_b = o3d.pipelines.registration.evaluate_registration(
                self.cloud_b, self.src_aligned, 0.5, np.eye(4))
            fit_f = float(ev_f.fitness)
            fit_b = float(ev_b.fitness)
            fit_avg = 0.5 * (fit_f + fit_b)
            
            del ev_f, ev_b
            gc.collect()

            self._set_status("Нормали...")
            ne = normal_error_metric(self.cloud_b, self.src_aligned, verbose=False)
            gc.collect()

            fsv_rate_f = fsv_rate_b = fsv_rate_avg = fsv_score = None
            if self.traj_a is not None and self.traj_b is not None:
                self._set_status("FSV forward...")
                fsv_f = evaluate_free_space_consistency(
                    cloud_source=self.src_aligned, cloud_reference=self.cloud_b,
                    trajectory_positions=self.new_traj_a[:, 1:4], trajectory_orientations=self.new_traj_a[:, 4:8],
                    tolerance=0.2, collision_radius=0.2, surface_margin=0.4, surface_knn_radius=0.4,
                    max_rays_per_pose=30, max_poses=100, pose_seed=42, ray_seed=42,
                    max_pairs_per_pose=30, max_violation_pairs=5000)
                fsv_rate_f = fsv_f["contradiction_rate"] / 100.0
                del fsv_f
                gc.collect()

                self._set_status("FSV backward...")
                fsv_b = evaluate_free_space_consistency(
                    cloud_source=self.cloud_b, cloud_reference=self.src_aligned,
                    trajectory_positions=self.traj_b[:, 1:4], trajectory_orientations=self.traj_b[:, 4:8],
                    tolerance=0.2, collision_radius=0.2, surface_margin=0.4, surface_knn_radius=0.4,
                    max_rays_per_pose=30, max_poses=100, pose_seed=42, ray_seed=42,
                    max_pairs_per_pose=30, max_violation_pairs=5000)
                fsv_rate_b = fsv_b["contradiction_rate"] / 100.0
                del fsv_b
                gc.collect()

                fsv_rate_avg = 0.5 * (fsv_rate_f + fsv_rate_b)
                fsv_score = 1.0 - fsv_rate_avg

            num = fit_avg + ne["normal_score"]
            den = 2.0
            if fsv_score is not None:
                num += fsv_score
                den += 1.0
            combined = num / den

            self.metrics = {
                "voxel_size": float(VOXEL_SIZE),
                "n_points_a_raw": int(self.n_a_raw),
                "n_points_b_raw": int(self.n_b_raw),
                "n_points_a": int(len(self.cloud_a.points)),
                "n_points_b": int(len(self.cloud_b.points)),
                "icp_fitness_fwd": fit_f,
                "icp_fitness_bwd": fit_b,
                "icp_fitness_avg": fit_avg,
                "normal_score": float(ne["normal_score"]),
                "normal_mean_angle_deg": float(ne.get("mean_angle_deg", float("nan"))),
                "normal_consistency": float(ne.get("consistency", 0.0)),
                "normal_n_common": int(ne.get("n_common", 0)),
                "fsv_rate_fwd": fsv_rate_f,
                "fsv_rate_bwd": fsv_rate_b,
                "fsv_rate_avg": fsv_rate_avg,
                "fsv_score": fsv_score,
                "combined": float(combined),
                "teaser_time_s": float(t_teaser),
            }

            self.root.after(0, self._update_metrics_display)
            self._set_status("Готово. Проверьте метрики и дайте вердикт.")
            self.root.after(0, lambda: self.btn_correct.config(state="normal"))
            self.root.after(0, lambda: self.btn_incorrect.config(state="normal"))
        except Exception as e:
            err = str(e)
            self._set_status(f"Ошибка: {err}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", err))
        finally:
            self.computing = False
            gc.collect()

    def _update_metrics_display(self):
        m = self.metrics
        def f(v):
            return "—" if v is None else f"{v:.4f}"

        lines = []
        lines.append(f"Cloud A: {self.cloud_a_path.get()}")
        lines.append(f"  raw={m['n_points_a_raw']}  →  voxel({m['voxel_size']})={m['n_points_a']} pts")
        lines.append(f"Cloud B: {self.cloud_b_path.get()}")
        lines.append(f"  raw={m['n_points_b_raw']}  →  voxel({m['voxel_size']})={m['n_points_b']} pts")
        lines.append("")
        lines.append("=== ICP fitness (max_corr=0.5 m) ===")
        lines.append(f"  forward  (A_aligned → B):  {f(m['icp_fitness_fwd'])}")
        lines.append(f"  backward (B → A_aligned):  {f(m['icp_fitness_bwd'])}")
        lines.append(f"  avg:                       {f(m['icp_fitness_avg'])}")
        lines.append("")
        lines.append("=== Normals ===")
        lines.append(f"  normal_score:      {f(m['normal_score'])}")
        lines.append(f"  mean_angle_deg:    {f(m['normal_mean_angle_deg'])}")
        lines.append(f"  consistency:       {f(m['normal_consistency'])}")
        lines.append(f"  n_common:          {m['normal_n_common']}")
        lines.append("")
        lines.append("=== FSV (free-space violation) ===")
        if m['fsv_rate_fwd'] is None:
            lines.append("  (пропущено — нет траекторий)")
        else:
            lines.append(f"  rate_fwd:      {f(m['fsv_rate_fwd'])}")
            lines.append(f"  rate_bwd:      {f(m['fsv_rate_bwd'])}")
            lines.append(f"  rate_avg:      {f(m['fsv_rate_avg'])}")
            lines.append(f"  fsv_score:     {f(m['fsv_score'])}")
        lines.append("")
        lines.append("=== Combined ===")
        lines.append(f"  combined:      {f(m['combined'])}")
        lines.append(f"  teaser_time_s: {m['teaser_time_s']:.2f}")
        lines.append("")
        lines.append("=== Transformation T ===")
        for row in self.T:
            lines.append("  " + "  ".join(f"{v: .6f}" for v in row))

        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", "\n".join(lines))

    def save_verdict(self, verdict):
        if not self.metrics:
            messagebox.showwarning("Нет метрик", "Сначала запустите TEASER + метрики")
            return
        out = self.output_path.get().strip()
        if not out:
            messagebox.showerror("Нет файла", "Укажите путь для сохранения")
            return

        record = {
            "timestamp": datetime.now().isoformat(),
            "cloud_a": self.cloud_a_path.get(),
            "cloud_b": self.cloud_b_path.get(),
            "traj_a": self.traj_a_path.get() or None,
            "traj_b": self.traj_b_path.get() or None,
            "T": self.T.tolist(),
            "metrics": self.metrics,
            "verdict": verdict,
        }

        try:
            d = os.path.dirname(os.path.abspath(out))
            if d:
                os.makedirs(d, exist_ok=True)
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.save_status.set(f"Сохранено ({verdict}) → {out}")
            messagebox.showinfo("OK", f"Вердикт '{verdict}' сохранён в:\n{out}")
        except Exception as e:
            messagebox.showerror("Ошибка записи", str(e))

    def reset_all(self):
        self.cloud_a = self.cloud_b = None
        self.n_a_raw = self.n_b_raw = 0
        self.traj_a = self.traj_b = None
        self.T = None
        self.src_aligned = None
        self.new_traj_a = None
        self.metrics = {}
        self._loaded_key = None
        self.metrics_text.delete("1.0", "end")
        self.btn_correct.config(state="disabled")
        self.btn_incorrect.config(state="disabled")
        self.save_status.set("")
        self.status_var.set("Сброшено. Выберите новые файлы.")
        gc.collect()


def main():
    root = tk.Tk()
    TeaserVerifyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()