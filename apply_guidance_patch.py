"""
apply_guidance_patch.py
=======================
Добавляет guidance в molopt_score_model_guided.py:

reward = - beta  * sim_penalty    # непохожа на референсы
         - gamma * pharma_penalty # но попадает в фармакофор

Запуск:
    python apply_guidance_patch.py
"""

from pathlib import Path

src  = Path("models/molopt_score_model_guided.py")
text = src.read_text()

OLD = "            ligand_pos = ligand_pos_next\n"

NEW = """            ligand_pos = ligand_pos_next

            # ── Guidance ──────────────────────────────────────────────────
            if guidance_scale > 0 and pos0_from_e is not None:
                try:
                    with torch.enable_grad():
                        pos0_grad = pos0_from_e.detach().requires_grad_(True)

                        # 1. Similarity penalty — толкаем ОТ референсов
                        sim_penalty = torch.tensor(0.0)
                        if encoder is not None and z_refs is not None:
                            try:
                                z_gen = encoder(
                                    torch.zeros(pos0_grad.shape[0],
                                                dtype=torch.long),
                                    pos0_grad,
                                    batch_ligand
                                )
                                sims = torch.stack([
                                    torch.nn.functional.cosine_similarity(
                                        z_gen, z_ref.unsqueeze(0)
                                    )
                                    for z_ref in z_refs
                                ])
                                sim_penalty = sims.max()
                            except Exception:
                                pass

                        # 2. Pharmacophore penalty — тянем К точкам кармана
                        pharma_penalty = torch.tensor(0.0)
                        if pharma_coords is not None:
                            try:
                                pc = pharma_coords.to(pos0_grad.device)
                                dists = torch.cdist(pos0_grad, pc)
                                pharma_penalty = dists.min(dim=1).values.mean()
                            except Exception:
                                pass

                        # 3. Итоговый reward
                        reward = (- beta  * sim_penalty
                                  - gamma * pharma_penalty)

                        if reward.requires_grad:
                            grad = torch.autograd.grad(
                                reward, pos0_grad,
                                retain_graph=False
                            )[0]
                            ligand_pos = (ligand_pos +
                                          guidance_scale * grad.detach())
                except Exception:
                    pass
            # ── End Guidance ───────────────────────────────────────────────

"""

assert OLD in text, "Target string not found!"
text = text.replace(OLD, NEW, 1)
src.write_text(text)
print("✓ Guidance patch applied to models/molopt_score_model_guided.py")
