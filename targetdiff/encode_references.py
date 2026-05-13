"""
encode_references.py
====================
Шаг 3: Кодирует референсные молекулы через обученный GNN энкодер.
        Сохраняет эмбеддинги для использования в guidance.

Входные данные:
    encoder/mol_encoder.pt
    ../small_molecules/reference_mol/P12931_references.smi

Выход:
    encoder/z_refs.pt   ← тензор эмбеддингов [N_refs, 64]

Запуск:
    python encode_references.py
    python encode_references.py \
        --refs ../small_molecules/reference_mol/P12931_references.smi \
        --encoder encoder/mol_encoder.pt
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import SchNet
from torch_scatter import scatter

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Пути ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
ENCODER_DIR = BASE_DIR / "encoder"

# ── Константы ──────────────────────────────────────────────────────────────
HIDDEN_DIM = 128
OUT_DIM    = 64
MAX_ATOMS  = 50


# ════════════════════════════════════════════════════════════════════════════
# 1. Энкодер (та же архитектура что в train_encoder.py)
# ════════════════════════════════════════════════════════════════════════════

class MoleculeEncoder(nn.Module):
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
# 2. SMILES → 3D
# ════════════════════════════════════════════════════════════════════════════

def smiles_to_3d(smiles: str):
    """SMILES → (pos, z) или None если не удалось."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    if mol.GetNumAtoms() > MAX_ATOMS:
        return None

    mol    = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if result != 0:
        result = AllChem.EmbedMolecule(mol, randomSeed=0)
    if result != 0:
        return None

    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

    conf = mol.GetConformer()
    pos  = torch.tensor(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=torch.float32
    )
    z = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long
    )
    return pos, z


# ════════════════════════════════════════════════════════════════════════════
# 3. Главная функция
# ════════════════════════════════════════════════════════════════════════════

def run(refs_path: Path, encoder_path: Path):

    print("=" * 55)
    print("  Encoding Reference Molecules")
    print(f"  Refs:    {refs_path}")
    print(f"  Encoder: {encoder_path}")
    print("=" * 55)

    # ── Загружаем энкодер ─────────────────────────────────────────────────
    log.info("Loading encoder...")
    ckpt    = torch.load(encoder_path, weights_only=False)
    encoder = MoleculeEncoder(
        hidden_dim = ckpt.get("hidden_dim", HIDDEN_DIM),
        out_dim    = ckpt.get("out_dim",    OUT_DIM),
    )
    encoder.load_state_dict(ckpt["model_state"])
    encoder.eval()

    log.info(f"Encoder loaded (epoch={ckpt['epoch']} "
             f"Pearson={ckpt.get('pearson', 'N/A')} "
             f"margin={ckpt['margin']})")

    # ── Загружаем референсные SMILES ──────────────────────────────────────
    log.info(f"Loading reference SMILES from {refs_path}...")
    smiles_list = [
        line.strip() for line in refs_path.read_text().split("\n")
        if line.strip()
    ]
    log.info(f"Found {len(smiles_list)} reference SMILES")

    # ── Кодируем каждую молекулу ──────────────────────────────────────────
    z_refs   = []
    valid    = []
    failed   = []

    with torch.no_grad():
        for i, smi in enumerate(smiles_list):
            mol_3d = smiles_to_3d(smi)

            if mol_3d is None:
                log.warning(f"  [{i+1}] Failed 3D embedding: {smi[:40]}")
                failed.append(smi)
                continue

            pos, z = mol_3d
            emb    = encoder(z, pos)  # [1, out_dim]
            z_refs.append(emb.squeeze(0))
            valid.append(smi)
            log.info(f"  [{i+1}/{len(smiles_list)}] ✓ encoded "
                     f"(norm={emb.norm().item():.3f})")

    if not z_refs:
        raise ValueError("No reference molecules could be encoded!")

    # ── Стекаем в тензор ──────────────────────────────────────────────────
    z_refs_tensor = torch.stack(z_refs)  # [N_refs, out_dim]

    # ── Сохраняем ─────────────────────────────────────────────────────────
    out_path = ENCODER_DIR / "z_refs.pt"
    torch.save({
        "z_refs":  z_refs_tensor,
        "smiles":  valid,
        "n_refs":  len(valid),
        "out_dim": z_refs_tensor.shape[1],
    }, out_path)

    print(f"\n{'='*55}")
    print(f"Done!")
    print(f"  Encoded:  {len(valid)} / {len(smiles_list)} molecules")
    print(f"  Failed:   {len(failed)}")
    print(f"  Shape:    {z_refs_tensor.shape}  [N_refs x out_dim]")
    print(f"  Saved →   {out_path}")
    print(f"{'='*55}")

    return z_refs_tensor


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode reference molecules with trained GNN encoder"
    )
    parser.add_argument(
        "--refs",
        type=str,
        default="../small_molecules/reference_mol/P12931_references.smi",
        help="Path to reference SMILES file"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="encoder/mol_encoder.pt",
        help="Path to trained encoder checkpoint"
    )
    args = parser.parse_args()

    run(Path(args.refs), Path(args.encoder))
