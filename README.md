# Guided Diffusion for Next-in-Class Molecule Generation
## Against Src Kinase (UniProt P12931)

Pharmacophore-constrained guided diffusion for generating next-in-class 
small molecule inhibitors of Src kinase using type-aware pharmacophore guidance.

---

## Key Results

| Metric | Baseline | Guided (scale=0.10) |
|--------|----------|---------------------|
| Validity | 28% | **97%** |
| QED | 0.320 | **0.367** |
| SAScore | 5.292 | **5.004** |
| MW (Da) | 494 | 416 |
| max Tanimoto | 0.140 | **0.135** |
| Pharma match | 0.409 | 0.389 |
| Vina best | -6.81 | **-7.95 kcal/mol** |

Optimal guidance_scale = **0.10** (best Vina score + best pharma_match among guided)

---

## Method

### Problem
Tanimoto similarity is non-differentiable through 3D coordinates:
```
∂Tanimoto(ECFP4(x), ECFP4(r)) / ∂x = does not exist
```

### Solution: Differentiable Tanimoto Proxy via SchNet Regression
```
f_θ : (z, pos) → R^64
cosine(f_θ(a), f_θ(b)) ≈ 2·Tanimoto(a,b) - 1
Pearson correlation = 0.980
```

### Guided Diffusion
```
reward = -β·sim_penalty - γ·pharma_penalty

sim_penalty    = max_r cosine(f_θ(x_0^pred), z_r)   # away from refs
pharma_penalty = type-aware distance to PLIP points   # toward pharmacophore

Type-aware:
  hbond points  → only N(7), O(8) atoms (< 3.5Å)
  hydrophobic   → only C(6), F(9), S(16), Cl(17) (< 4.0Å)

Applied only at t < 200 (low noise regime)
```

### Grid Search Results (guidance_scale)

| Scale | Validity | QED | Pharma match | Vina best |
|-------|----------|-----|--------------|-----------|
| Baseline | 28% | 0.320 | 0.409 | -6.81 |
| 0.01 | 96% | 0.409 | 0.348 | -7.40 |
| 0.05 | 96% | 0.357 | 0.377 | -7.04 |
| **0.10** | **97%** | 0.367 | **0.389** | **-7.95** |
| 0.20 | 96% | 0.379 | 0.373 | -6.97 |

---

## Quick Start

### 1. Clone repository
```bash
git clone https://github.com/KurbanovDavid777/guided-diffusion-src.git
cd guided-diffusion-src/targetdiff
```

### 2. Install dependencies
```bash
conda create -n targetdiff python=3.8
conda activate targetdiff
pip install torch==2.4.1 torch-geometric==2.6.1 torch-scatter
pip install rdkit biopython requests pandas scipy scikit-learn easydict pyyaml vina
```

### 3. Download pretrained models
```bash
python -c "
import requests, tarfile, os
url = 'https://zenodo.org/api/records/14041881/files/targetdiff_pretrained_models.tar.gz/content'
r = requests.get(url, stream=True, allow_redirects=True)
open('/tmp/models.tar.gz', 'wb').write(r.content)
tarfile.open('/tmp/models.tar.gz').extractall('.')
os.remove('/tmp/models.tar.gz')
print('Done!')
"
```

### 4. Apply guidance patch (once)
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
cp models/molopt_score_model.py models/molopt_score_model_guided.py
python apply_guidance_patch_v2.py
```

### 5. Run baseline
```bash
python scripts/sample_for_pocket.py configs/sampling_35.yml \
    --pdb_path ../small_molecules/pockets/1O49_493_pocket10.pdb \
    --result_path outputs/1O49_baseline_35 \
    --num_samples 100 \
    --device cuda
```

### 6. Run guided generation (grid search)
```bash
for scale in 0.01 0.05 0.10 0.20; do
    python sample_guided.py configs/sampling_35.yml \
        --pdb_path ../small_molecules/pockets/1O49_493_pocket10.pdb \
        --result_path outputs/1O49_guided_scale_${scale} \
        --num_samples 100 \
        --beta 0.5 \
        --gamma 0.3 \
        --guidance_scale ${scale} \
        --guidance_start_step 200 \
        --plip_path ../small_molecules/coordinates/1O49_493_coords.json \
        --device cuda
done
```

### 7. Convert and compare
```bash
# Convert SDF to CSV
for scale in 0.01 0.05 0.10 0.20; do
    python sdf_to_csv.py --run 1O49_guided_scale_${scale}
done

# Compare all scales with pharmacophore match metric
python compare_all_scales.py \
    --baseline results/1O49_baseline_35.csv \
    --scales 0.01 0.05 0.10 0.20 \
    --refs ../small_molecules/reference_mol/P12931_references.smi \
    --plip_path ../small_molecules/coordinates/1O49_493_coords.json
```

### 8. Docking
```bash
# Create receptor
obabel ../small_molecules/target/1O49_target.pdb \
    -O receptor.pdbqt -xr --partialcharge gasteiger

# Dock all
for scale in 0.01 0.05 0.10 0.20; do
    python run_docking.py --run 1O49_guided_scale_${scale}
done
```

---

## Project Structure

```
├── Dockerfile
├── docker-compose.yml
├── README.md
│
├── small_molecules/                    # Data preparation
│   ├── fetch_structures.py             # Download PDB structures
│   ├── data_prep.py                    # Reference molecules from ChEMBL
│   ├── plip_analysis.py                # Pharmacophore analysis
│   ├── select_target.py                # Best structure selection
│   ├── extract_coordinates.py          # Pocket coordinates
│   ├── extract_pockets.py              # Pocket extraction
│   ├── build_encoder_dataset_v2.py     # Dataset (ChEMBL negatives)
│   ├── pockets/                        # Extracted pockets
│   ├── coordinates/                    # PLIP pharmacophore coords
│   └── reference_mol/                  # 21 reference inhibitors (holdout)
│
└── targetdiff/                         # Generation pipeline
    ├── prepare_conformations.py         # 3D conformations cache
    ├── train_encoder.py                 # GNN encoder (Tanimoto regression)
    ├── encode_references.py             # Encode reference molecules
    ├── apply_guidance_patch_v2.py       # Type-aware guidance patch
    ├── sample_guided.py                 # Guided generation
    ├── sdf_to_csv.py                    # Convert SDF to CSV
    ├── compare_all_scales.py            # Compare guidance_scale experiments
    ├── run_docking.py                   # AutoDock Vina
    ├── encoder/                         # Trained encoder + z_refs
    ├── configs/                         # Sampling configs
    └── models/                          # TargetDiff model code
```

---

## Encoder Training Details

- **Architecture**: SchNet (4 blocks, hidden=128) + MLP projection → z[64]
- **Dataset**: ChEMBL P12931 actives (pChEMBL ≥ 7) + ChEMBL P12931 weak (pChEMBL < 5)
- **Loss**: MSE regression on Tanimoto (not binary classification)
- **Metric**: Pearson correlation = **0.980**
- **Key fix**: negatives from same target (ChEMBL P12931) — no chemotype shortcut

---

## References

1. Guan et al. TargetDiff: 3D Equivariant Diffusion for Target-Aware Molecule Generation. ICLR 2023.
2. Schütt et al. SchNet: A continuous-filter convolutional neural network. NeurIPS 2017.
3. Dhariwal & Nichol. Diffusion Models Beat GANs on Image Synthesis. NeurIPS 2021.
4. Salentin et al. PLIP: fully automated protein-ligand interaction profiler. NAR 2015.
5. Morris et al. AutoDock Vina. J. Comput. Chem. 2010.
