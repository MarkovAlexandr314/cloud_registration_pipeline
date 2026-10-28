from pathlib import Path
import yaml


def load_config(config_path):
    """Читает YAML, разрешает относительные пути относительно самого конфига."""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base = config_path.parent

    def _abs(p: str) -> str:
        p = Path(p).expanduser()
        return str(p if p.is_absolute() else (base / p))

    # Обязательные скалярные поля
    for key in ("teaser_bin", "checkpoint_dir", "output_dir"):
        if key not in cfg or not cfg[key]:
            raise ValueError(f"В конфиге отсутствует обязательное поле '{key}'")
        cfg[key] = _abs(cfg[key])

    if not Path(cfg["teaser_bin"]).exists():
        raise FileNotFoundError(f"TEASER-бинарник не найден: {cfg['teaser_bin']}")
    if not Path(cfg["checkpoint_dir"]).is_dir():
        raise FileNotFoundError(f"checkpoint_dir не существует: {cfg['checkpoint_dir']}")
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Сессии
    if not cfg.get("sessions"):
        raise ValueError("Список 'sessions' пуст")

    sessions = []
    seen_ids = set()
    for s in cfg["sessions"]:
        for k in ("id", "cloud", "traj"):
            if k not in s:
                raise ValueError(f"В сессии отсутствует поле '{k}': {s}")
        sid = int(s["id"])
        if sid in seen_ids:
            raise ValueError(f"Дублирующийся id сессии: {sid}")
        seen_ids.add(sid)

        cloud = _abs(s["cloud"])
        traj  = _abs(s["traj"])
        if not Path(cloud).exists():
            raise FileNotFoundError(f"[id={sid}] облако не найдено: {cloud}")
        if not Path(traj).exists():
            raise FileNotFoundError(f"[id={sid}] траектория не найдена: {traj}")

        sessions.append({
            "id":    sid,
            "cloud": cloud,
            "traj":  traj,
            "label": s.get("label") or Path(cloud).name,
        })
    cfg["sessions"] = sessions

    # Значения по умолчанию для вложенных секций
    cfg.setdefault("teaser", {})
    cfg.setdefault("merge",  {})
    cfg["teaser"].setdefault("voxel",       0.3)
    cfg["teaser"].setdefault("noise_bound", 0.5)
    cfg["merge"].setdefault("prob_threshold",    0.5)
    cfg["merge"].setdefault("fsv_rate_hard_max", 0.95)
    cfg["merge"].setdefault("fsv_max_poses",     100)
    cfg["merge"].setdefault("fsv_max_rays",      30)
    cfg["merge"].setdefault("max_iterations",    15)

    return cfg