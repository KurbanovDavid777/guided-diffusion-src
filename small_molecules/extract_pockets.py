# extract_pockets.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select

class PocketSelect(Select):
    def __init__(self, center, cutoff=10.0):
        self.center = np.array(center)
        self.cutoff = cutoff

    def accept_residue(self, residue):
        for atom in residue:
            dist = np.linalg.norm(atom.coord - self.center)
            if dist < self.cutoff:
                return True
        return False

# Читаем топ-10 карманов из нашего CSV
df = pd.read_csv("coordinates/P12931_pocket_centers.csv")
df_top10 = df.nlargest(10, 'n_interactions')

# Парсим белок один раз
parser = PDBParser(QUIET=True)

# Папка для карманов
pockets_dir = Path("pockets")
pockets_dir.mkdir(exist_ok=True)

for _, row in df_top10.iterrows():
    pdb_id    = row['pdb_id']
    ligand_id = row['ligand_id']
    center    = [row['center_x'], row['center_y'], row['center_z']]

    # Берём PDB файл этой структуры
    pdb_path = Path(f"structures/P12931/{pdb_id}.pdb")

    if not pdb_path.exists():
        print(f"  {pdb_id}: PDB not found, skipping")
        continue

    structure = parser.get_structure(pdb_id, str(pdb_path))

    # Вырезаем карман
    out_path = pockets_dir / f"{pdb_id}_{ligand_id}_pocket10.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), PocketSelect(center, cutoff=10.0))

    print(f"  ✓ {pdb_id}_{ligand_id}: center={center} → {out_path}")

print(f"\nDone! {len(list(pockets_dir.glob('*.pdb')))} pockets saved → pockets/")