from plip.structure.preparation import PDBComplex

pdb = "/Users/davidkurbanov/Desktop/PROJECTS/small_molecules/structures/P12931/1O49.pdb"
mol = PDBComplex()
mol.load_pdb(pdb)
mol.analyze()

for key, site in mol.interaction_sets.items():
    inter = site.all_itypes
    if not inter:
        continue
    print(f"=== Ligand site: {key} ===")
    print("H-bonds:", len(site.hbonds_pdon + site.hbonds_ldon))
    print("Hydrophobic:", len(site.hydrophobic_contacts))
    # пример одного H-bond
    for hb in (site.hbonds_pdon + site.hbonds_ldon)[:3]:
        print(f"  HB: resnr={hb.resnr} restype={hb.restype} "
              f"protein_atom_coords={hb.a.coords if hasattr(hb,'a') else '?'}")
    break
