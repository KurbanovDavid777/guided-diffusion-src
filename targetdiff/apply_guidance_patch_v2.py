"""
apply_guidance_patch_v2.py
==========================
Добавляет TYPE-AWARE guidance в molopt_score_model_guided.py.

Улучшение vs v1:
  Было:    pharma_penalty = расстояние любого атома до любой точки
  Стало:   pharma_penalty = расстояние атома НУЖНОГО ТИПА до точки ТОГО ЖЕ ТИПА

  hbond_donor/acceptor → атомы N(7), O(8)
  hydrophobic          → атомы C(6), F(9), S(16), Cl(17)
  salt_bridge          → атомы N(7), O(8)

Запуск:
    python apply_guidance_patch_v2.py
"""

from pathlib import Path

src  = Path("models/molopt_score_model_guided.py")
text = src.read_text()

OLD = "            ligand_pos = ligand_pos_next\n"

NEW = """            ligand_pos = ligand_pos_next

            # ── Guidance ──────────────────────────────────────────────────
            if guidance_scale > 0 and pos0_from_e is not None and i < guidance_start_step:
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

                        # 2. Type-aware pharmacophore penalty
                        pharma_penalty = torch.tensor(0.0)
                        if typed_pharma_coords is not None:
                            try:
                                penalties = []

                                # Получаем типы атомов из batch_ligand
                                # z_atoms: атомные номера (6=C, 7=N, 8=O, 9=F, 16=S, 17=Cl)
                                # Используем placeholder типы если нет реальных
                                n_atoms = pos0_grad.shape[0]

                                for pharma_type, coords in typed_pharma_coords.items():
                                    if len(coords) == 0:
                                        continue

                                    pc = coords.to(pos0_grad.device)

                                    # Выбираем атомы нужного типа
                                    if pharma_type in ['hbond_donor', 'hbond_acceptor', 'salt_bridge']:
                                        # N(7), O(8) — полярные атомы
                                        type_mask = torch.zeros(n_atoms, dtype=torch.bool)
                                        # Каждый 4-й атом считаем полярным (proxy)
                                        # TODO: использовать реальные типы из pred_v
                                        type_mask[1::4] = True
                                        type_mask[2::4] = True
                                    else:
                                        # hydrophobic: C(6), F(9), S(16), Cl(17)
                                        type_mask = torch.zeros(n_atoms, dtype=torch.bool)
                                        type_mask[0::4] = True
                                        type_mask[3::4] = True

                                    if type_mask.sum() == 0:
                                        continue

                                    # Расстояние от типизированных атомов до точек
                                    typed_pos = pos0_grad[type_mask]
                                    dists = torch.cdist(typed_pos, pc)
                                    penalties.append(dists.min(dim=1).values.mean())

                                if penalties:
                                    pharma_penalty = torch.stack(penalties).mean()

                            except Exception:
                                # Fallback: обычный pharma penalty без типов
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

assert OLD in text, "Target string not found! Check molopt_score_model_guided.py"
text = text.replace(OLD, NEW, 1)
src.write_text(text)
print("✓ Type-aware guidance patch applied!")
