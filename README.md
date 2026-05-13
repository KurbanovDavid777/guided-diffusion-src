# Guided Diffusion for Next-in-Class Molecule Generation
## Against Src Kinase (UniProt P12931)

Pharmacophore-constrained guided diffusion for generating next-in-class 
small molecule inhibitors of Src kinase.

---

## Quick Start

### 1. Build Docker image
```bash
docker-compose build
```
> Первый build ~20 минут (скачивает PyTorch, PyG, pretrained models)

### 2. Run container
```bash
docker-compose run guided-diffusion
```

### 3. Inside container — run experiments

**Baseline generation (no guidance):**
```bash
cd /workspace/targetdiff
export PYTHONPATH=$(pwd):$PYTHONPATH

python scripts/sample_for_pocket.py configs/sampling_35.yml \
    --pdb_path ../small_molecules/pockets/1O49_493_pocket10.pdb \
    --result_path outputs/1O49_baseline_35 \
    --num_samples 100 \
    --device cuda  # или cpu если нет GPU
```

**Guided generation:**
```bash
python sample_guided.py configs/sampling_35.yml \
    --pdb_path ../small_molecules/pockets/1O49_493_pocket10.pdb \
    --result_path outputs/1O49_guided_35_v2 \
    --num_samples 100 \
    --beta 0.5 \
    --gamma 0.3 \
    --guidance_scale 0.05 \
    --guidance_start_step 200 \
    --plip_path ../small_molecules/coordinates/1O49_493_coords.json \
    --device cuda
```

**Convert SDF to CSV:**
```bash
python sdf_to_csv.py --run 1O49_baseline_35
python sdf_to_csv.py --run 1O49_guided_35_v2
```

**Compare molecules:**
```bash
python compare_molecules.py \
    --baseline results/1O49_baseline_35.csv \
    --guided   results/1O49_guided_35_v2.csv \
    --refs     ../small_molecules/reference_mol/P12931_references.smi
```

**Docking:**
```bash
# Convert receptor
obabel ../small_molecules/target/1O49_target.pdb \
    -O receptor.pdbqt -xr --partialcharge gasteiger

python run_docking.py --run 1O49_baseline_35
python run_docking.py --run 1O49_guided_35_v2
```

---

## Project Structure

```
├── Dockerfile
├── docker-compose.yml
├── README.md
│
├── small_molecules/           # Data preparation
│   ├── fetch_structures.py    # Download PDB structures
│   ├── data_prep.py           # Reference molecules from ChEMBL
│   ├── plip_analysis.py       # Pharmacophore analysis
│   ├── select_target.py       # Best structure selection
│   ├── extract_coordinates.py # Pocket coordinates
│   ├── extract_pockets.py     # Pocket extraction
│   ├── build_encoder_dataset_v2.py  # Dataset for encoder
│   ├── pockets/               # Extracted pockets
│   ├── coordinates/           # PLIP pharmacophore coords
│   └── reference_mol/         # 21 reference inhibitors
│
└── targetdiff/                # Generation pipeline
    ├── prepare_conformations.py  # 3D conformations cache
    ├── train_encoder.py          # GNN encoder training
    ├── encode_references.py      # Encode reference molecules
    ├── sample_guided.py          # Guided generation
    ├── apply_guidance_patch.py   # Patch TargetDiff
    ├── sdf_to_csv.py             # Convert SDF to CSV
    ├── compare_molecules.py      # Compare metrics
    ├── run_docking.py            # AutoDock Vina
    ├── configs/                  # Sampling configs
    ├── models/                   # TargetDiff model code
    ├── scripts/                  # Sampling scripts
    └── encoder/                  # Trained encoder checkpoints
```

---

## Method Overview

### Problem
Tanimoto similarity is non-differentiable through 3D coordinates:
```
∂Tanimoto(ECFP4(x), ECFP4(r)) / ∂x = does not exist
```

### Solution
Train a differentiable Tanimoto proxy via SchNet regression:
```
f_θ : (z, pos) → R^64
cosine(f_θ(a), f_θ(b)) ≈ 2·Tanimoto(a,b) - 1
Pearson correlation = 0.980
```

### Guided Diffusion
```
reward = -β·sim_penalty - γ·pharma_penalty

sim_penalty   = max_r cosine(f_θ(x_0^pred), z_r)  # away from refs
pharma_penalty = mean_i min_p ||atom_i - p||_2      # toward pharmacophore

Applied only at t < 200 (low noise regime)
```

### Results
| Metric | Baseline | Guided |
|--------|----------|--------|
| Validity | 28% | 99% |
| QED | 0.320 | 0.355 |
| SAScore | 5.289 | 4.966 |
| Vina best | -6.81 | -7.53 kcal/mol |

---

## Requirements

- Docker + docker-compose
- NVIDIA GPU recommended (CUDA 11.8+)
- ~10GB disk space

---

## References

1. Guan et al. TargetDiff: 3D Equivariant Diffusion for Target-Aware Molecule Generation. ICLR 2023.
2. Schütt et al. SchNet: A continuous-filter convolutional neural network. NeurIPS 2017.
3. Dhariwal & Nichol. Diffusion Models Beat GANs on Image Synthesis. NeurIPS 2021.
4. Salentin et al. PLIP: fully automated protein-ligand interaction profiler. NAR 2015.
