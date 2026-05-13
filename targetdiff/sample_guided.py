"""
sample_guided.py
================
Шаг 4+5: Генерация молекул с guidance внутри диффузии.

Guidance:
  + affinity  → тянем к активным молекулам
  - similarity → толкаем от референсных ингибиторов

Входные данные:
    configs/sampling.yml
    encoder/mol_encoder.pt
    encoder/z_refs.pt
    pockets/{name}_pocket10.pdb  ← карман мишени

Выход:
    outputs/{run_name}/sdf/*.sdf

Запуск:
    python sample_guided.py \
        --pdb_path ../small_molecules/pockets/1O49_493_pocket10.pdb \
        --result_path outputs/1O49_guided \
        --num_samples 100 \
        --alpha 1.0 \
        --beta 0.5 \
        --guidance_scale 0.05 \
        --device cpu
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.transforms import Compose
from torch_scatter import scatter

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model_guided import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
from utils.data import PDBProtein
from utils import reconstruct
from rdkit import Chem

# ── Пути ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
ENCODER_DIR = BASE_DIR / "encoder"

# ── Энкодер (та же архитектура) ───────────────────────────────────────────
from torch_geometric.nn import SchNet

class MoleculeEncoder(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=64):
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

    def forward(self, z, pos, batch=None):
        if batch is None:
            batch = torch.zeros(len(z), dtype=torch.long, device=z.device)
        h = self.schnet.embedding(z)
        edge_index, edge_weight = self.schnet.interaction_graph(pos, batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)
        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        h_mol = scatter(h, batch, dim=0, reduce='mean')
        return F.normalize(self.proj(h_mol), dim=-1)


def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element':      torch.empty([0, ], dtype=torch.long),
            'pos':          torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index':   torch.empty([2, 0], dtype=torch.long),
            'bond_type':    torch.empty([0, ], dtype=torch.long),
        }
    )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',          type=str,
                        default='configs/sampling.yml')
    parser.add_argument('--pdb_path',      type=str, required=True)
    parser.add_argument('--result_path',   type=str, required=True)
    parser.add_argument('--num_samples',   type=int, default=100)
    parser.add_argument('--device',        type=str, default='cpu')
    parser.add_argument('--batch_size',    type=int, default=100)
    # Guidance параметры
    parser.add_argument('--alpha',          type=float, default=1.0,
                        help='Affinity guidance weight')
    parser.add_argument('--beta',           type=float, default=0.5,
                        help='Similarity guidance weight')
    parser.add_argument('--gamma',          type=float, default=0.3,
                        help='Pharmacophore guidance weight')
    parser.add_argument('--guidance_start_step', type=int, default=200,
                        help='Apply guidance only when t < this value')
    parser.add_argument('--guidance_scale', type=float, default=0.05,
                        help='Guidance step size')
    parser.add_argument('--encoder_path',  type=str,
                        default='encoder/mol_encoder.pt')
    parser.add_argument('--z_refs_path',   type=str,
                        default='encoder/z_refs.pt')
    parser.add_argument('--plip_path',     type=str,
                        default='../small_molecules/coordinates/1O49_493_coords.json',
                        help='Path to PLIP coordinates JSON')
    args = parser.parse_args()

    # ── Конфиг ───────────────────────────────────────────────────────────
    config = misc.load_config(args.config)
    if args.num_samples:
        config.sample.num_samples = args.num_samples

    logger = misc.get_logger('evaluate')
    logger.info(config)

    # ── Загружаем модель ─────────────────────────────────────────────────
    ckpt = torch.load(config.model.checkpoint,
                      map_location=args.device)
    logger.info(f'Training Config: {ckpt["config"]}')

    # Трансформации
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode   = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer  = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([protein_featurizer])

    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(
        ckpt['model'],
        strict=False if 'train_config' in config.model else True
    )
    logger.info(f'Successfully load the model! {config.model.checkpoint}')
    encoder = None
    z_refs  = None

    if Path(args.encoder_path).exists() and args.guidance_scale > 0:
        logger.info(f'Loading encoder from {args.encoder_path}...')
        enc_ckpt = torch.load(args.encoder_path, weights_only=False)
        encoder  = MoleculeEncoder(
            hidden_dim=enc_ckpt.get('hidden_dim', 128),
            out_dim   =enc_ckpt.get('out_dim',    64),
        )
        encoder.load_state_dict(enc_ckpt['model_state'])
        encoder.eval()
        logger.info(f'Encoder loaded (Pearson={enc_ckpt.get("pearson", "N/A")})')
    else:
        logger.warning('Encoder not found — running without similarity guidance')

    if Path(args.z_refs_path).exists() and args.guidance_scale > 0:
        logger.info(f'Loading z_refs from {args.z_refs_path}...')
        refs_data = torch.load(args.z_refs_path, weights_only=False)
        z_refs    = refs_data['z_refs']  # [N_refs, 64]
        logger.info(f'Loaded {refs_data["n_refs"]} reference embeddings')
    else:
        logger.warning('z_refs not found — running without similarity guidance')

    # ── Загружаем фармакофорные координаты из PLIP ───────────────────────
    pharma_coords = None
    raw_coords    = []
    if Path(args.plip_path).exists() and args.guidance_scale > 0:
        import json
        logger.info(f'Loading pharmacophore coords from {args.plip_path}...')
        plip_data  = json.load(open(args.plip_path))
        raw_coords = plip_data.get('raw_coords', [])
        if raw_coords:
            pharma_coords = torch.tensor(raw_coords, dtype=torch.float32)
            logger.info(f'Loaded {len(raw_coords)} pharmacophore points')
    else:
        logger.warning('PLIP coords not found — no pharmacophore guidance')

    # ── Передаём guidance параметры в модель ──────────────────────────────
    model.guidance_scale = args.guidance_scale
    model.guidance_start_step = args.guidance_start_step
    model.beta           = args.beta
    model.gamma          = args.gamma
    model.encoder        = encoder
    model.z_refs         = z_refs
    model.pharma_coords  = pharma_coords
    model.affinity_model = None

    logger.info(f'Guidance: scale={args.guidance_scale} '
                f'beta={args.beta} gamma={args.gamma}')
    logger.info(f'Encoder:  {"ON" if encoder else "OFF"}')
    logger.info(f'z_refs:   {"ON (" + str(len(z_refs)) + " refs)" if z_refs is not None else "OFF"}')
    logger.info(f'Pharma:   {"ON (" + str(len(raw_coords)) + " points)" if pharma_coords is not None else "OFF"}')

    # ── Загружаем карман ──────────────────────────────────────────────────
    data = pdb_to_pocket_data(args.pdb_path)
    data = transform(data)

    # ── Генерация ─────────────────────────────────────────────────────────
    os.makedirs(args.result_path, exist_ok=True)

    all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, \
    all_pred_v0_traj, all_pred_vt_traj, time_list = \
        sample_diffusion_ligand(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size,
            device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms
        )

    logger.info('Sample done!')

    # ── Реконструкция молекул ─────────────────────────────────────────────
    result = {
        'pred_pos':      all_pred_pos,
        'pred_v':        all_pred_v,
        'pred_pos_traj': all_pred_pos_traj,
        'pred_v_traj':   all_pred_v_traj,
        'pred_v0_traj':  all_pred_v0_traj,
        'pred_vt_traj':  all_pred_vt_traj,
        'time':          time_list,
        'data':          data,
    }
    torch.save(result, os.path.join(args.result_path, 'sample.pt'))

    sdf_dir = os.path.join(args.result_path, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)

    n_recon = 0
    n_complete = 0

    for i, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
        try:
            # Правильная конвертация индексов → атомные номера
            pred_atom_type = trans.get_atomic_number_from_index(
                pred_v, mode=ligand_atom_mode
            )
            pred_aromatic = trans.is_aromatic_from_index(
                pred_v, mode=ligand_atom_mode
            )
            mol = reconstruct.reconstruct_from_generated(
                pred_pos, pred_atom_type, pred_aromatic
            )
            mol_frags = Chem.rdmolops.GetMolFrags(mol,
                                                   asMols=True,
                                                   sanitizeFrags=False)
            mol = max(mol_frags, default=mol,
                      key=lambda m: m.GetNumAtoms())
            n_recon += 1
            try:
                Chem.SanitizeMol(mol)
                n_complete += 1
                Chem.MolToMolFile(
                    mol,
                    os.path.join(sdf_dir, f'{i:03d}.sdf')
                )
            except Exception:
                pass
        except Exception:
            pass

    logger.info(f'Reconstruction done!')
    logger.info(f'n recon: {n_recon} n complete: {n_complete}')
    logger.info(f'Results are saved in {args.result_path}')


if __name__ == '__main__':
    main()
