import torch
import torch.nn as nn
from pathlib import Path

FEATURE_COLS = [
    "icp_fitness_fwd", "icp_fitness_bwd",
    "normal_score", "normal_mean_angle_deg",
    "fsv_rate_fwd", "fsv_rate_bwd",
]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 25), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(25, 10),         nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(10, 1),
        )
    def forward(self, x):
        return self.net(x)
_MODEL_CACHE = {}


def load_final(ckpt_dir):
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


def predict_final(raw_rows, ckpt_dir):
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