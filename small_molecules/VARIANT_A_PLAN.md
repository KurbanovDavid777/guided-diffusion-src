# 🧬 Вариант А — Guided Diffusion для генерации молекул
## Classifier Guidance внутри TargetDiff

> **Цель:** направить диффузионную генерацию одновременно  
> к **активным** молекулам (↑ affinity) и **новым** структурам (↓ similarity)

---

## 🎯 Идея в одной строке

```
x_{t-1} = denoise(x_t) + α·∇affinity(x_0_pred) - β·∇similarity(x_0_pred, refs)
```

На каждом шаге диффузии добавляем градиент который **тянет** молекулу  
в сторону высокой аффинности и **толкает** от известных ингибиторов.

---

## ✅ Что уже готово

| Файл | Описание |
|------|----------|
| `structures/P12931/*.pdb` | 31 PDB структура киназы Src |
| `target/1O49_target.pdb` | Лучшая мишень (20 взаимодействий, 1.7Å) |
| `reference_mol/P12931_references.smi` | 50 референсных SMILES из ChEMBL |
| `coordinates/1O49_493_coords.json` | Координаты кармана для TargetDiff |
| `plip_results/` | PLIP отчёты (48 binding sites) |

---

## 📋 План реализации

### Шаг 1 — Расширение датасета для энкодера
**Время:** ~1 день  
**Кто:** человек отвечающий за данные

**Проблема:** 50 референсов → ~1225 пар → мало для обучения энкодера  
**Решение:** расширяем до ~100k пар

```
50 референсов (P12931, pChEMBL ≥ 6)        ← уже есть
+ ~5000 молекул P12931 (pChEMBL ≥ 5)       ← тянем из ChEMBL
+ ~10000 случайных drug-like из ZINC        ← негативные примеры
──────────────────────────────────────────
→ ~100k пар (похожие / непохожие)
→ сохраняем в encoder_dataset/
```

**Пары для обучения:**
- Похожие: Tanimoto > 0.6 → близко в эмбеддинг пространстве
- Непохожие: Tanimoto < 0.3 → далеко в эмбеддинг пространстве

---

### Шаг 2 — Обучение GNN энкодера
**Время:** ~2-3 дня  
**Кто:** человек отвечающий за генеративку

**Архитектура:** SchNet (PyTorch Geometric) — работает с 3D координатами

```python
MoleculeEncoder:
  SchNet(hidden_dim=256)     # обрабатывает 3D атомы
  → Linear(256 → 128)        # проекция
  → L2 normalize             # нормализация для cosine similarity
```

**Loss:** NT-Xent (InfoNCE) — contrastive learning

```
Вход:   пары молекул (anchor, positive, negative)
        каждая в 3D (SMILES → RDKit EmbedMolecule → xyz)
Выход:  энкодер который знает что похоже а что нет
```

**Чекпоинт сохраняем в:** `encoder/mol_encoder.pt`

---

### Шаг 3 — Кодирование референсов
**Время:** ~2 часа (после шага 2)  
**Кто:** любой

```python
# Один раз перед генерацией:
z_refs = encode_references(
    smiles_list = open("reference_mol/P12931_references.smi").readlines(),
    encoder     = load("encoder/mol_encoder.pt"),
    device      = "cuda"
)
# z_refs — список из 50 эмбеддингов размером 128
# сохраняем в encoder/z_refs.pt
```

---

### Шаг 4 — Интеграция guidance в TargetDiff
**Время:** ~2-3 дня  
**Кто:** человек отвечающий за генеративку

**Что менять в TargetDiff:**

```
models/diffusion.py
  функция p_sample()  ← здесь стандартный денойзинг
        ↓
  заменяем на guided_step():
    1. x_0_pred, x_prev = p_sample(x_t, pocket, t)  # стандартный шаг
    2. aff  = affinity_head(x_0_pred, pocket)         # аффинность
    3. z    = encoder(x_0_pred)                        # эмбеддинг
    4. sim  = max(cosine(z, z_refs))                   # схожесть
    5. grad = ∂(α·aff - β·sim) / ∂x_t                # градиент
    6. x_prev = x_prev + scale · grad                 # guidance
```

**Ключевые параметры (тюнинг на валидации):**

| Параметр | Начальное значение | Что делает |
|----------|-------------------|------------|
| `alpha` | 1.0 | вес affinity (к активным) |
| `beta` | 0.5 | вес similarity (от референсов) |
| `guidance_scale` | 0.05 | размер шага guidance |

---

### Шаг 5 — Генерация с guidance
**Время:** ~1 день  
**Кто:** человек отвечающий за генеративку

```bash
python generate_guided.py \
    --protein  target/1O49_target.pdb \
    --pocket   coordinates/1O49_493_coords.json \
    --refs     encoder/z_refs.pt \
    --encoder  encoder/mol_encoder.pt \
    --n_samples 500 \
    --alpha 1.0 \
    --beta  0.5 \
    --out   generated/
```

**Выход:**
```
generated/
├── molecules.sdf      ← 3D позы всех молекул
├── molecules.csv      ← SMILES + affinity (из TargetDiff)
└── valid_smiles.smi   ← только валидные SMILES
```

---

### Шаг 6 — Скоринг сгенерированных молекул
**Время:** ~2 дня  
**Кто:** человек отвечающий за скоринг

```python
# Для каждой молекулы из generated/molecules.csv:

affinity  = targetdiff_affinity(mol, pocket)   # уже есть из генерации
tanimoto  = max_tanimoto(mol, ref_smiles)       # vs P12931_references.smi
qed       = rdkit_qed(mol)                     # drug-likeness
sa_score  = rdkit_sascore(mol)                 # синтезируемость
admet     = admet_ai_predict(mol)              # токсичность

# Итоговый скор:
score = 0.45 * norm(affinity)
      + 0.30 * (1 - tanimoto)   # novelty
      + 0.15 * qed
      + 0.10 * norm(sa_score)
```

**Фильтры:**
```
affinity  < -7 ккал/моль  (или pKd > 7)
tanimoto  < 0.4            next-in-class
QED       > 0.5
SAScore   < 4.0
ADMET     pass
```

---

### Шаг 7 — Population RL (итеративная оптимизация)
**Время:** ~3-5 дней  
**Кто:** все вместе

```
Итерация 1:
  generate_with_guidance() → 500 молекул
  scoring() → reward для каждой
  топ-50 по reward → fine-tune TargetDiff
        ↓
Итерация 2:
  TargetDiff (улучшенный) → 500 молекул
  scoring() → reward
  топ-50 → fine-tune
        ↓
... повторяем 10-20 итераций
```

**Ожидаемый прогресс:**
```
Итерация 1:  случайные молекулы, avg reward ~0.3
Итерация 5:  начинает улучшаться, avg reward ~0.5
Итерация 10: хорошие молекулы, avg reward ~0.65+
```

---

### Шаг 8 — Финальный анализ
**Время:** ~2 дня  
**Кто:** LLM команда

```
Топ-20 молекул по reward
        ↓
Qwen LLM:
  SureChEMBL API → патентный анализ по InChIKey
  Молекулярный паспорт (механизм, офф-таргеты)
        ↓
Финальный отчёт + Streamlit UI
```

---

## 🗓️ Таймлайн

```
Неделя 1:
  Пн-Вт  →  Шаг 1: расширение датасета
  Ср-Пт  →  Шаг 2: обучение энкодера

Неделя 2:
  Пн     →  Шаг 3: кодирование референсов
  Вт-Чт  →  Шаг 4: интеграция в TargetDiff
  Пт     →  Шаг 5: первая генерация

Неделя 3:
  Пн-Вт  →  Шаг 6: скоринг
  Ср-Пт  →  Шаг 7: RL итерации (1-10)

Неделя 4:
  Пн-Вт  →  Шаг 7: RL итерации (10-20)
  Чт-Пт  →  Шаг 8: патентный анализ + UI
```

---

## 👥 Распределение по команде

| Роль | Задачи |
|------|--------|
| Данные (ты) | Шаг 1 (датасет), референсы уже готовы |
| Генеративка | Шаги 2, 3, 4, 5 (энкодер + TargetDiff) |
| Докинг/RL | Шаги 6, 7 (скоринг + RL петля) |
| LLM + UI | Шаг 8 (Qwen + Streamlit) |

---

## ⚠️ Риски и митигация

| Риск | Влияние | Митигация |
|------|---------|-----------|
| Guidance ломает структуру молекул | Много невалидных SMILES | Уменьшить guidance_scale до 0.01-0.05 |
| Энкодер плохо обучился | Плохой сигнал similarity | Проверить на валидации до интеграции |
| RL не сходится | Слабые молекулы | Начать с Population RL без fine-tune |
| alpha/beta сложно тюнить | Нестабильная генерация | Grid search по 9 комбинациям α∈{0.5,1,2} β∈{0.3,0.5,1} |

---

## 📦 Финальные артефакты

```
generated/
├── top_molecules.csv      ← топ-20 молекул со всеми скорами
├── top_molecules.sdf      ← 3D позы в кармане
└── patent_report.json     ← Qwen анализ патентоспособности

encoder/
├── mol_encoder.pt         ← обученный GNN энкодер
└── z_refs.pt              ← эмбеддинги референсов
```

---

## 🏆 Критерии успеха

```
✓ Топ-5 молекул имеют pKd > 7 (affinity лучше чем Dasatinib ~8.0)
✓ Tanimoto < 0.4 для всех топ молекул (next-in-class)
✓ SAScore < 4.0 у ≥ 80% молекул (синтезируемы)
✓ Хотя бы 3 молекулы без патентных хитов в SureChEMBL
✓ Полный цикл (генерация → отчёт) < 2 часов на GPU
```

---

*ПОРА ПОБЕЖДАТЬ* 🚀
