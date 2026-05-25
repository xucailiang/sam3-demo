# Point Protocol Fix: Centroid → Nearest-Foreground Point

> 日期：2026-05-25
> 状态：CrackForest 完成，DeepCrack 待跑

---

## 1. 问题

原代码 `experiments/prompt_generator.py` 使用 `cv2.connectedComponentsWithStats(...).centroids` 计算每个 GT 连通域的**几何质心**作为 point prompt。但凹形、分叉、C/U 形裂缝的质心可能落在背景像素上。

**实测数据**（分析脚本位于本文档末尾）：

| 数据集 | 总连通域 | 质心落背景 | 受影响图像 |
|--------|---------|-----------|-----------|
| CrackForest | 316 | 171 (54.1%) | 112/118 (94.9%) |
| DeepCrack | 1970 | 794 (40.3%) | 425/537 (79.1%) |

CrackForest 超过一半的 point prompt 指在背景上，这解释了 point 单模式极低的原始结果（IoU 0.1166）。

---

## 2. 代码修改

### 修改文件

`experiments/prompt_generator.py` — 仅修改 `generate_point_prompts` 方法

### 核心逻辑

```python
for each connected component:
    cx, cy = centroids[i]  # geometric centroid (float)
    ix, iy = int(round(cx)), int(round(cy))

    if gt_mask[iy, ix] == 255 and labels[iy, ix] == i:
        # Centroid already on foreground → use directly
        point = (ix, iy)
    else:
        # Find nearest foreground pixel in this component via Euclidean distance
        fg_coords = cv2.findNonZero(component_mask)
        nearest = argmin(distance(fg_coords, centroid))
        point = nearest
```

### 修改前后对比

| 项目 | 旧 (centroid) | 新 (nearest-fg) |
|------|-------------|----------------|
| 函数签名 | 不变 | 不变 |
| 返回格式 | `([(x,y),...], [1,...])` | 不变 |
| 标注定位 | 几何质心 | 保证落在裂缝前景像素上 |
| 论文表述 | "foreground point" (不准确) | "nearest-foreground point per component" (准确) |

---

## 3. 测试验证

### 3.1 现有测试

```bash
.venv/bin/python -m pytest experiments/tests/test_prompt_generator.py -v
```

6 个测试全部通过。Hypothesis 属性测试用 blob 样本，质心天然在前景上，nearest-fg 逻辑透明（无行为变化）。

### 3.2 新增手工验证

```python
# U-shape crack — 质心落背景 (40.0, 51.4)，新代码选 nearest fg (40, 68)
old_centroid = (40.0, 51.4)  # on background
new_point    = (40.0, 68.0)  # on foreground ✓
```

### 3.3 测试覆盖盲区（未修复）

- `test_points_match_components` 仍只用 blob 样本，无法覆盖凹形裂缝
- 建议后续增加 U 形/C 形连通域的确定性测试（但本次未加）

---

## 4. 实验结果

### 执行命令

```bash
.venv/bin/python experiments/rerun_point_experiments.py --dataset crackforest
```

耗时：18.1 分钟 (RTX 3090, 24 张 × 10 组)

### 完整对比表 (CrackForest, 24 test images)

| Method | Old (centroid) | New (nearest-fg) | Δ | Change |
|--------|---------------|-----------------|-----|--------|
| **point single** | **0.1166** | **0.3014** | **+0.1848** | **+158.4%** |
| ampf | 0.4250 | 0.4145 | -0.0106 | -2.5% |
| no_detection | 0.4362 | 0.4087 | -0.0274 | -6.3% |
| no_stability | 0.4348 | 0.4279 | -0.0069 | -1.6% |
| no_boundary | 0.4215 | 0.4122 | -0.0093 | -2.2% |
| no_alignment | 0.2911 | 0.3506 | +0.0595 | +20.5% |
| fusion_max | 0.4276 | 0.4084 | -0.0192 | -4.5% |
| **fusion_mean** | **0.4535** | **0.4587** | **+0.0052** | **+1.1%** |
| text+point | 0.4257 | 0.3972 | -0.0285 | -6.7% |
| box+point | 0.4075 | 0.4012 | -0.0064 | -1.6% |

**Unchanged baselines (for reference):**

| Method | IoU |
|--------|-----|
| text | 0.4355 |
| box | 0.3439 |

### 关键发现

1. **Point 单独提升 2.58 倍** — 从灾难性的 0.1166 提升到 0.3014，证明 centroid 落背景是旧 point 低性能的主要原因之一

2. **Point 仍弱于 text 和 box** — 即使前景点保证，point (0.3014) < box (0.3439) < text (0.4355)，说明 SAM3 点提示对裂缝几何的适配性确实有限

3. **更好的 point 反而损害融合** — 9 个融合组中 7 个下降（-1.6% 到 -6.7%），只有 fusion_mean (+1.1%) 和 no_alignment (+20.5%) 提升。no_alignment 提升是因为更好的 point 单独 mask 减少了随机错位；但 confidence-weighted AMPF 中，更好的 point 被给予了错误的高置信度权重，反而污染了融合结果

4. **论文核心论点被加强** — confidence-weighted fusion 的 calibration 问题在新 point 下更加明显；simple mean fusion 仍然是最鲁棒的融合策略；alignment 仍然是关键组件（no_alignment v2: 0.3506 vs fusion_mean v2: 0.4587, gap 仍然巨大）

---

## 5. 结果文件路径

### 代码

| 文件 | 说明 |
|------|------|
| `experiments/prompt_generator.py` | 修改后的 point 生成代码 |
| `experiments/rerun_point_experiments.py` | 针对性重跑脚本 |

### 新结果 CSV（v2, nearest-foreground point）

```
experiments/results/
├── crackforest_point_v2.csv                  # point single mode
├── crackforest_ampf_v2.csv                   # AMPF full
├── crackforest_ablation_no_detection_v2.csv  # ablation: w/o S_det
├── crackforest_ablation_no_stability_v2.csv  # ablation: w/o S_stab
├── crackforest_ablation_no_boundary_v2.csv   # ablation: w/o S_bnd
├── crackforest_ablation_no_alignment_v2.csv  # ablation: w/o alignment
├── crackforest_ablation_fusion_max_v2.csv    # ablation: fusion = max
├── crackforest_ablation_fusion_mean_v2.csv   # ablation: fusion = mean
├── crackforest_combo_text+point_v2.csv       # combo: text + point
├── crackforest_combo_box+point_v2.csv        # combo: box + point
└── point_experiments_v2.csv                  # 汇总 (240 rows)
```

### 旧结果 CSVs（centroid point, 用于对比）

```
experiments/results/
├── crackforest_point.csv
├── crackforest_ampf.csv
├── crackforest_ablation_*.csv
├── crackforest_combo_*.csv
└── all_results.csv                           # DeepCrack only (650 rows)
```

---

## 6. 待完成

- [ ] **DeepCrack 重跑** — 237 张图 × 10 组，预计 4-5 小时 (RTX 3090)
- [ ] **更新 manuscript.md 结果表** — 用 v2 数据替换旧 point/融合结果
- [ ] **更新 Discussion** — "centroid-point protocol" → "nearest-foreground point protocol"，并纳入新发现（更好的 point 仍弱、更损害 confidence-weighted fusion）
- [ ] **U 形裂缝测试** — 在 `test_prompt_generator.py` 中增加凹形连通域确定性测试

---

## 附录：质心分析脚本

```python
# 统计 centroid 落背景比例
import cv2, numpy as np
from experiments.data_loader import CrackDataset

ds = CrackDataset('crackforest', 'experiments/datasets/CrackForest-dataset')

total_components = 0
centroid_on_bg = 0

for idx in range(len(ds)):
    _, gt_mask = ds[idx]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        gt_mask, connectivity=8
    )
    for i in range(1, num_labels):
        total_components += 1
        cx, cy = centroids[i]
        ix, iy = int(round(cx)), int(round(cy))
        ix = max(0, min(ix, gt_mask.shape[1] - 1))
        iy = max(0, min(iy, gt_mask.shape[0] - 1))
        if gt_mask[iy, ix] == 0:
            centroid_on_bg += 1

print(f'{centroid_on_bg}/{total_components} = {100*centroid_on_bg/total_components:.1f}%')
# Output: 171/316 = 54.1%  (CrackForest)
# Output: 794/1970 = 40.3% (DeepCrack)
```
