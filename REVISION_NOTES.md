# Manuscript Revision Notes — 2026-05-25

本文档记录基于导师预审意见 (`manuscript_review_issues.md`) 对 `manuscript.md` 和 `EXPERIMENTS.md` 进行的第一轮系统性修正。该轮修正为文档表述修正，未重新运行实验。

> **后续更新（2026-05-26）**：point protocol 已从 centroid-only 修正为 nearest-foreground point，并完成 CrackForest 与 DeepCrack 重跑。最终论文与实验记录应以 `POINT_PROTOCOL_FIX.md`、`experiments/results/all_results_v2.csv`、`experiments/results/summary_stats_v2.csv` 和 `experiments/results/paper_tables_v2.tex` 为准。本文件保留为第一轮修订历史记录。

---

## 修正总览

导师评审的核心判断：论文最有价值的是做一个 protocol-aware empirical evaluation，即诚实地说明哪些结果是 deployable 的、哪些是 GT-assisted diagnostic 的。修正的目标是将论文定位从"提出强 AMPF 方法"调整为"SAM3 多提示模式的 protocol-aware 系统性评估"。

---

## 1. Metric 命名: `mIoU` → `Foreground IoU`

### 问题
代码中 IoU 计算为 per-image foreground crack IoU 的均值，不是 semantic segmentation 论文中常见的 class-averaged mIoU（包含 background class）。审稿人可能误解数值含义。

### 修正
- 所有表格表头和正文中的 `mIoU` → `IoU`（表格）或 `foreground IoU` / `mean foreground IoU`（正文）
- Section 3.1 和 Section 3.5 明确说明: "per-image foreground crack IoU averaged over the test set, rather than class-averaged mIoU including background"
- 唯一保留 `mIoU` 的位置是 Section 3.5 的解释句，用来说明本报告的 metric 与 class-averaged mIoU 的区别

### 影响
纯表述修正，数值不变。

---

## 2. Point Prompt 表述: `foreground point` → `centroid-derived point`

### 问题
- 代码 `experiments/prompt_generator.py` 使用 `cv2.connectedComponentsWithStats(...).centroids` 计算几何质心
- 质心不保证落在裂缝前景像素上（凹形、分叉、C/U 形裂缝的质心可能落在背景）
- 测试 `test_points_match_components` 使用小圆 blob 作为样本，不能覆盖真实裂缝几何
- 论文写 "one foreground point per connected crack component" 不准确

### 修正
- Section 3.1 点提示定义: "one foreground point" → "one centroid-derived positive point"
- Section 3.2 Protocol 表: Box/Point interpretation 从 "Ground-truth-assisted spatial prompt upper bound" → "GT-derived single-box and centroid-point diagnostic protocol"
- Section 3.3 点生成段落: 完整改写，明确 centroid 限制
- Section 5.1 结果分析: 加入 "may reflect both the geometry of cracks and limitations of centroid-derived prompts"
- Section 6.2: 完整改写，提出 open question（nearest-foreground / skeleton-based / multi-point 是否更好）
- Section 7: 新增第 2 条 Limitations，明确 centroid protocol 限制
- Section 8 Conclusion: 加入 centroid protocol 限定

### 影响
纯表述修正，数值不变。这是导师指出的最严重的 factual error——原表述可能被审稿人质疑为 point prompt 的不公平测试。

---

## 3. GT-Assisted Alignment 边界: 移除/弱化 `upper bound` 定位

### 问题
- AMPF/mean fusion 的 alignment 使用 GT instance 与候选 mask 的 IoU 匹配
- 这意味着 AMPF/mean fusion 不是 fully automatic deployment method
- 摘要中 "best overall mIoU" 未加限定，容易误导审稿人

### 修正
- Abstract: "best overall mIoU" → "under the GT-assisted alignment protocol, achieves the highest diagnostic foreground IoU"
- Section 3.2: "upper-bound or diagnostic protocols" → "diagnostic protocols...rather than as oracle upper bounds"
- 所有 Section 5-8 中 mean fusion 的结果陈述均加入 "under the GT-assisted alignment protocol" 限定
- Section 5.4 表格保留 Protocol 列，明确区分 `automatic_text_prompt` 和 `oracle_gt_prompt_gt_alignment`
- Section 6.4: "best mIoU" → "highest diagnostic foreground IoU"
- Section 8: 结论从 "mean fusion performs best" 调整为 "the fusion results show that aligned multi-prompt information contains value, but the current alignment relies on GT-assisted instance matching and should be viewed as a diagnostic analysis"
- 全文移除 "upper bound" 描述（box/point 协议本身偏弱，不宜称 upper bound）

### 影响
纯表述修正，数值不变。这是避免 desk reject 的关键修正——如果不加限定，审稿人发现 alignment 使用 GT 后会质疑诚信。

---

## 4. 监督基线 Backbone/Epochs/Optimizer 描述修正

### 问题
- `EXPERIMENTS.md` 写 U-Net 用 EfficientNet-b0、DeepLabV3+ 用 ResNet-50、统一 100 epochs、AdamW
- 代码 `experiments/run_baselines.py` 实际: 两者均用 `resnet34`、默认 50 epochs (U-Net/DeepLabV3+)、`torch.optim.Adam`（非 AdamW）
- YOLOv8-seg 实际 100 epochs，文档中统一写 100 不准确

### 修正
- `manuscript.md` Section 4.2: 补充准确描述 — ResNet-34 backbone, ImageNet pretrained, 50 epochs, Adam, BCE+Dice loss
- `EXPERIMENTS.md` Section 3.3: 标题改为 "U-Net/DeepLabV3+: 50 epochs, Adam, lr=1e-4; YOLOv8-seg: 100 epochs"
- Backbone: EfficientNet-b0 → ResNet-34; ResNet-50 → ResNet-34
- 参数量更新: ~5.7M → ~24M, 保持 ~27M

### 影响
纯文档修正。实验结果来自 ResNet-34/50-epochs/Adam 配置，数值不变。论文定位为 "reference supervised baselines rather than exhaustively tuned SOTA" 也与此一致。

---

## 5. Text Prompt 描述统一
6. 路径引用修正

### 问题
- `EXPERIMENTS.md` protocol 表写 text prompt = "crack, fracture, fissure, break"
- 代码 `prompt_generator.py` 实际: text single-mode baseline 只用 `"crack"`
- `fracture/fissure/break` 只在 AMPF stability scoring 的 synonym perturbation 中使用

### 修正
- `EXPERIMENTS.md` Section 3.1 表格: "crack, fracture, fissure, break" → `"crack"`
- 增加 note: "同义词仅用于 AMPF text-prompt stability scoring，不用于 text single-mode baseline"

### 影响
纯文档修正。

---

## 6. Data and Code Availability 路径修正

### 问题
- 论文引用 `sam3-demo/backend/...`、`sam3-demo/experiments/...` 路径
- 实际项目根目录就是 `sam3-demo/`，不存在嵌套的 `sam3-demo/` 子目录
- `sam3-demo/experiments/` 目录实际不存在
- 测试文件只存在于 `experiments/` 下，没有 mirror

### 修正
- `sam3-demo/backend/app/services/...` → `backend/app/services/...`
- 移除 "mirrored under sam3-demo/experiments/" 描述
- 移除不存在的 `sam3-demo/experiments/tests/...` 测试路径引用

### 影响
纯路径修正，无代码变更。

---

## 修正后的论文表述一致性

| 概念 | 旧表述 | 新表述 |
|------|--------|--------|
| Metric | `mIoU` | `IoU` (表格) / `foreground IoU` (正文) |
| Point prompt | `foreground point` | `centroid-derived positive point` |
| Box/Point protocol | `oracle upper bound` | `GT-derived diagnostic protocol` |
| Mean fusion 定位 | `best overall` | `highest under GT-assisted alignment` (diagnostic) |
| 监督基线 backbone | EfficientNet-b0 / ResNet-50 | ResNet-34 (两者) |
| 监督基线 epochs | 100 (统一) | 50 (U-Net/DeepLabV3+), 100 (YOLOv8-seg) |
| 监督基线 optimizer | AdamW | Adam |

---

## 未修改但需注意的事项

以下为导师建议中**未纳入本次修正**的项目，属于"强烈建议"或"后续工作":

1. **Nearest-foreground point protocol**: 已完成，见 `POINT_PROTOCOL_FIX.md` 与 v2 结果文件
2. **Per-instance box protocol**: 需修改 box 生成逻辑 + 重跑 box 相关实验 (~2h)
3. **Qualitative mask visualization**: 需生成 overlay 图（可从已有推理结果生成，不需重跑）
4. **Dice/F1 去重**: 当前二值像素级设置下 Dice=F1，主表可去掉 F1 放入 appendix
5. **Threshold sensitivity analysis**: text confidence 0.25, fusion confidence 0.5, IoU matching 0.1 的敏感度分析
6. **Runtime/memory profiling**: 各 prompt mode 的耗时和显存分析
7. **Automatic proposal protocol**: 用 image processing / edge detection 替代 GT 生成 prompt，是 AMPF 从 diagnostic 走向 deployable 的关键

---

## 修正文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `manuscript.md` | 已修正 | 所有 "Must Fix" 项 |
| `EXPERIMENTS.md` | 已修正 | Text prompt / 监督基线 / IoU naming |
| `manuscript_review_issues.md` | 未修改 | 导师原评审，保留作为参考 |
| `experiments/results/all_results_v2.csv` | 已生成 | nearest-foreground point protocol 后的最终结果文件 |
