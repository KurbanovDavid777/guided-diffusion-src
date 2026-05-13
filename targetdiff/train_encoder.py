"""
train_encoder.py (v2 — Tanimoto Regression)
============================================
Обучает GNN энкодер молекул через регрессию на Tanimoto.

Изменения vs v1:
  - Loss: MSE(cosine(z_a, z_b), 2*Tanimoto-1) вместо NT-Xent
  - Метрика: Pearson + Spearman корреляция вместо AUC-ROC
  - Target: реальный Tanimoto [0,1] из кэша вместо label {0,1}

Математика:
  cosine(z_a, z_b) ∈ [-1, +1]
  Tanimoto ∈ [0, 1]
  target = 2*Tanimoto - 1 ∈ [-1, +1]  ← масштабируем в диапазон cosine
  loss = MSE(cosine(z_a, z_b), target)

Входные данные:
  encoder_dataset_v2/conformations_8000_20000.pt  ← кэш с Tanimoto

Выход:
  encoder/mol_encoder.pt    ← чекпоинт энкодера
  encoder/training_log.csv  ← лог обучения

Запуск:
    python train_encoder.py \
        --dataset_dir ../small_molecules/encoder_dataset_v2 \
        --max_similar 8000 \
        --max_dissimilar 20000 \
        --epochs 30 \
        --patience 5
"""

import argparse
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import SchNet
from torch_scatter import scatter

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Пути ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
ENCODER_DIR = BASE_DIR / "encoder"
ENCODER_DIR.mkdir(exist_ok=True)

# ── Константы ──────────────────────────────────────────────────────────────
HIDDEN_DIM = 128
OUT_DIM    = 64


# ════════════════════════════════════════════════════════════════════════════
# 1. Dataset — загружает кэш с реальным Tanimoto
# ════════════════════════════════════════════════════════════════════════════

class TanimotoDataset(Dataset):
    """
    Загружает кэш конформаций.
    Каждый элемент содержит реальный Tanimoto для регрессии.
    """

    def __init__(self, dataset_dir: Path,
                 max_similar: int, max_dissimilar: int):

        cache_path = (dataset_dir /
                      f"conformations_{max_similar}_{max_dissimilar}.pt")

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found: {cache_path}\n"
                f"Run prepare_conformations.py first!"
            )

        log.info(f"Loading cache from {cache_path}...")
        self.data = torch.load(cache_path, weights_only=False)

        # Проверяем что в кэше есть tanimoto а не label
        if "tanimoto" not in self.data[0]:
            raise ValueError(
                "Cache contains 'label' instead of 'tanimoto'!\n"
                "Delete cache and re-run prepare_conformations.py"
            )

        random.shuffle(self.data)
        log.info(f"Loaded {len(self.data)} pairs ✓")

        # Статистика распределения Tanimoto
        tanimotos = [item["tanimoto"] for item in self.data]
        log.info(f"Tanimoto distribution: "
                 f"mean={np.mean(tanimotos):.3f} "
                 f"min={np.min(tanimotos):.3f} "
                 f"max={np.max(tanimotos):.3f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return batch


# ════════════════════════════════════════════════════════════════════════════
# 2. Энкодер (та же архитектура SchNet)
# ════════════════════════════════════════════════════════════════════════════

class MoleculeEncoder(nn.Module):
    """SchNet GNN энкодер молекул."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM,
                 out_dim: int = OUT_DIM):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=4,
            num_gaussians=50,
            cutoff=10.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor, pos: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(len(z), dtype=torch.long, device=z.device)

        h = self.schnet.embedding(z)
        edge_index, edge_weight = self.schnet.interaction_graph(pos, batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)

        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h_mol = scatter(h, batch, dim=0, reduce='mean')
        z_out = self.proj(h_mol)
        return F.normalize(z_out, dim=-1)


# ════════════════════════════════════════════════════════════════════════════
# 3. Tanimoto Regression Loss
# ════════════════════════════════════════════════════════════════════════════

def tanimoto_regression_loss(z_a: torch.Tensor,
                              z_b: torch.Tensor,
                              tanimoto: float) -> torch.Tensor:
    """
    MSE loss между cosine similarity и целевым Tanimoto.

    Математика:
      cosine(z_a, z_b) ∈ [-1, +1]
      Tanimoto ∈ [0, 1]
      target = 2 * Tanimoto - 1 ∈ [-1, +1]
      loss = MSE(cosine, target)

    Примеры:
      Tanimoto=0.0 → target=-1.0 (максимально далеко)
      Tanimoto=0.5 → target=0.0  (нейтрально)
      Tanimoto=1.0 → target=+1.0 (максимально близко)
    """
    sim    = F.cosine_similarity(z_a, z_b, dim=-1)
    target = torch.tensor(
        [2.0 * tanimoto - 1.0],
        dtype=torch.float32,
        device=z_a.device
    )
    return F.mse_loss(sim, target)


# ════════════════════════════════════════════════════════════════════════════
# 4. Оценка качества — Pearson + Spearman корреляция
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, device) -> dict:
    """
    Считает Pearson и Spearman корреляцию
    между cosine(z_a, z_b) и реальным Tanimoto.

    Это правильная метрика для регрессии:
      corr=1.0  → идеальное соответствие
      corr=0.0  → нет связи (случайно)
      corr=-1.0 → обратная связь (плохо)
    """
    model.eval()
    val_losses   = []
    cosine_sims  = []
    tanimotos    = []

    with torch.no_grad():
        for batch in loader:
            for item in batch:
                try:
                    z_a = item["z_a"].to(device)
                    pos_a = item["pos_a"].to(device)
                    z_b = item["z_b"].to(device)
                    pos_b = item["pos_b"].to(device)
                    tan = item["tanimoto"]

                    emb_a = model(z_a, pos_a)
                    emb_b = model(z_b, pos_b)

                    sim = F.cosine_similarity(emb_a, emb_b).item()
                    cosine_sims.append(sim)
                    tanimotos.append(tan)

                    loss = tanimoto_regression_loss(emb_a, emb_b, tan)
                    val_losses.append(loss.item())

                except Exception:
                    continue

    # Pearson и Spearman корреляция
    if len(cosine_sims) < 2:
        return {"val_loss": 0.0, "pearson": 0.0, "spearman": 0.0}

    cos_arr = np.array(cosine_sims)
    tan_arr = np.array(tanimotos)

    pearson  = pearsonr(cos_arr, tan_arr)[0]
    spearman = spearmanr(cos_arr, tan_arr)[0]

    # Дополнительная статистика
    mean_cos_high = cos_arr[tan_arr > 0.6].mean() if (tan_arr > 0.6).any() else 0.0
    mean_cos_low  = cos_arr[tan_arr < 0.3].mean() if (tan_arr < 0.3).any() else 0.0

    return {
        "val_loss":     float(np.mean(val_losses)),
        "pearson":      round(float(pearson),  4),
        "spearman":     round(float(spearman), 4),
        "mean_cos_high_tan": round(float(mean_cos_high), 4),
        "mean_cos_low_tan":  round(float(mean_cos_low),  4),
        "margin": round(float(mean_cos_high - mean_cos_low), 4),
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. Обучение
# ════════════════════════════════════════════════════════════════════════════

def train(dataset_dir: Path, max_similar: int, max_dissimilar: int,
          epochs: int = 30, batch_size: int = 32,
          lr: float = 1e-3, patience: int = 5):

    # SchNet требует CPU — radius операция не поддерживает MPS
    device = torch.device("cpu")
    log.info(f"Device: {device} (SchNet requires CPU)")

    # ── Датасет ───────────────────────────────────────────────────────────
    dataset = TanimotoDataset(dataset_dir, max_similar, max_dissimilar)

    n_val   = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    log.info(f"Train: {n_train} | Val: {n_val}")

    # ── Модель ────────────────────────────────────────────────────────────
    model     = MoleculeEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"Loss: Tanimoto MSE Regression")
    log.info(f"Metric: Pearson + Spearman correlation")

    # ── Цикл обучения ─────────────────────────────────────────────────────
    log_records    = []
    best_pearson   = -1.0
    best_val_loss  = float("inf")
    no_improve     = 0

    for epoch in range(1, epochs + 1):

        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0)
            valid = 0

            for item in batch:
                try:
                    emb_a = model(item["z_a"], item["pos_a"])
                    emb_b = model(item["z_b"], item["pos_b"])
                    tan   = item["tanimoto"]

                    loss = tanimoto_regression_loss(emb_a, emb_b, tan)
                    batch_loss = batch_loss + loss
                    valid += 1
                except Exception:
                    continue

            if valid > 0:
                (batch_loss / valid).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append((batch_loss / valid).item())

        optimizer.step()
        scheduler.step()

        # Валидация
        metrics    = evaluate(model, val_loader, device)
        train_loss = np.mean(train_losses) if train_losses else 0.0

        log.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train={train_loss:.4f} | "
            f"val={metrics['val_loss']:.4f} | "
            f"Pearson={metrics['pearson']:.3f} | "
            f"Spearman={metrics['spearman']:.3f} | "
            f"margin={metrics['margin']:.3f}"
        )

        log_records.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            **metrics
        })

        # Сохраняем лучший по Pearson корреляции
        if metrics["pearson"] > best_pearson:
            best_pearson  = metrics["pearson"]
            best_val_loss = metrics["val_loss"]
            no_improve    = 0
            ckpt_path     = ENCODER_DIR / "mol_encoder.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    metrics["val_loss"],
                "pearson":     metrics["pearson"],
                "spearman":    metrics["spearman"],
                "margin":      metrics["margin"],
                "hidden_dim":  HIDDEN_DIM,
                "out_dim":     OUT_DIM,
            }, ckpt_path)
            log.info(f"  ✓ Best model saved "
                     f"(Pearson={best_pearson:.3f})")
        else:
            no_improve += 1
            log.info(f"  No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            log.info(f"Early stopping at epoch {epoch}")
            break

    # ── Лог и итог ────────────────────────────────────────────────────────
    log_path = ENCODER_DIR / "training_log.csv"
    pd.DataFrame(log_records).to_csv(log_path, index=False)

    print(f"\n{'='*55}")
    print(f"Training complete!")
    print(f"  Best Pearson:  {best_pearson:.4f}")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint:    {ENCODER_DIR}/mol_encoder.pt")
    print(f"  Log:           {log_path}")
    print(f"\nPearson интерпретация:")
    print(f"  0.0  = нет корреляции (плохо)")
    print(f"  0.5  = умеренная корреляция")
    print(f"  0.7  = хорошая корреляция ✓")
    print(f"  0.9  = отличная корреляция ✓✓")
    print(f"{'='*55}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GNN encoder with Tanimoto regression"
    )
    parser.add_argument("--dataset_dir",    type=str, required=True)
    parser.add_argument("--max_similar",    type=int, default=8000)
    parser.add_argument("--max_dissimilar", type=int, default=20000)
    parser.add_argument("--epochs",         type=int, default=30)
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--patience",       type=int, default=5)
    args = parser.parse_args()

    train(
        dataset_dir    = Path(args.dataset_dir),
        max_similar    = args.max_similar,
        max_dissimilar = args.max_dissimilar,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        patience       = args.patience,
    )
