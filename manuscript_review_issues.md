# Manuscript Review Issues and Revision Plan

本文档从博士生导师/论文预审角度，按 `sam3-demo/` 作为提交项目根目录，梳理论文稿件与实验代码、测试和结果之间的主要不一致、论文风险和建议修订做法。

## 总体判断

当前稿件有投稿潜力，但不建议直接提交。论文最可靠的价值不是证明 AMPF 显著优于单一提示，而是做一个 protocol-aware empirical evaluation：在裂缝分割任务中，SAM3 text prompt 已经是很强的 deployable baseline，而复杂多提示融合只有在 GT-assisted alignment 下才表现出有限诊断价值。

因此，建议将论文定位从“提出一个强 AMPF 方法”调整为：

> A protocol-aware evaluation of SAM3 prompt modes and fusion strategies for crack segmentation.

核心修订目标是避免把 GT-assisted diagnostic results 误写成可部署算法性能，并修正 point prompt、IoU 指标、路径和 baseline 描述中的不一致。

## 1. Point-Based Protocol 与说明没有完全对上

### 问题

论文写法容易让读者理解为：

- 每个 GT 连通裂缝组件生成一个 foreground point；
- point 是一个较强的 oracle point prompt；
- point-based 低性能主要说明 SAM3 point prompt 不适合裂缝。

但代码实际做法是：

- `experiments/prompt_generator.py` 使用 `cv2.connectedComponentsWithStats(...).centroids` 计算每个连通域的几何质心；
- 质心不保证落在裂缝前景像素上；
- 对弯曲、分叉、环形、C/U 形或稀疏细长裂缝，质心可能落在背景区域。

因此，当前 point protocol 更准确地说是：

> one centroid-derived positive point per GT connected component

而不是严格的：

> one foreground point per connected crack component

### 测试覆盖问题

`experiments/tests/test_prompt_generator.py` 中 `test_points_match_components` 声称验证 “Point prompts match connected components”，并断言生成点落在前景上。但测试样本主要是小圆 blob，质心天然在前景内，不能覆盖真实裂缝中常见的凹形、弯曲、细长结构。

因此，该测试不能证明“任意裂缝连通域生成的点都是 foreground point”。

### 论文风险

当前 point 结果偏低，可能混合了两类原因：

1. SAM3 point prompt 对细长裂缝结构本身不够适配；
2. 当前 centroid point 生成协议太弱，部分点可能落在背景或非代表性区域。

如果不澄清，审稿人可能指出：这不是 point prompt 的公平 oracle upper bound，而是一个过弱的点生成策略。

### 当前 manuscript 中需要改的具体表述

如果 `manuscript.md` 随 `sam3-demo/` 一起作为提交稿件，以下 point mode 表述应统一替换。

1. **Task definition 中的 point prompt 定义**
   - 当前问题表述：`one foreground point per connected crack component`
   - 建议改为：`one centroid-derived positive point per GT connected component`

2. **Prompt protocol 表中的 Box/Point interpretation**
   - 当前问题表述：`Ground-truth-assisted spatial prompt upper bound`
   - 建议改为：`GT-derived single-box and centroid-point diagnostic protocol`

3. **Prompt generation 段落**
   - 当前问题表述：`The centroid of each component becomes one foreground point prompt.`
   - 建议改为：`The centroid of each component is used as one positive point prompt. Because this centroid is not guaranteed to lie on crack pixels for concave, branching, or sparse elongated components, this point protocol is treated as a centroid-point diagnostic baseline rather than an oracle foreground-point upper bound.`

4. **Results/Discussion 中的结论**
   - 当前风险：直接写 “point prompting is weak for elongated crack structures” 容易把低性能完全归因于 SAM3 point prompt。
   - 建议改为：`Under the centroid-point protocol used here, point prompting performs poorly on elongated crack structures. This may reflect both the geometry of cracks and the limitations of centroid-derived single-point prompts.`

### 建议改法

#### 论文表述修改

将 point prompt 相关表述从：

> one foreground point per connected crack component

改为：

> one centroid-derived positive point per GT connected component

并在 Methods 或 Limitations 中补充：

> The centroid is not guaranteed to lie on crack pixels for concave, branching, or sparse elongated components. Therefore, the point protocol should be interpreted as a simple centroid-point diagnostic baseline rather than an oracle point upper bound.

#### 代码改进选项

建议至少做其中一个：

1. **Nearest foreground point to centroid**
   - 计算连通域质心；
   - 如果质心四舍五入后不是前景像素，则选择离质心最近的前景像素；
   - 这样论文可以继续说 “foreground point”，但仍需说明是 nearest-to-centroid foreground point。

2. **Skeleton-based point**
   - 对裂缝连通域提取 skeleton；
   - 在 skeleton 上选择中心点、最长路径中点或多个代表点；
   - 更符合细长裂缝几何。

3. **Multi-point prompt**
   - 每个连通域生成端点、中点、分叉点或等距采样点；
   - 评估 single centroid point vs multi-point prompt；
   - 这会更公平地回答 “point prompting 是否真的不适合裂缝”。

#### 测试改进

增加至少一个凹形或 U/C 形连通域测试：

- 构造一个连通裂缝 mask，使几何质心落在背景；
- 验证旧 centroid 策略会失败；
- 如果改为 nearest foreground point，则验证生成点在前景上，并且仍属于对应连通域。

## 2. IoU / mIoU 指标命名不够严谨

### 问题

`experiments/evaluation.py` 中的 IoU 计算为：

```python
intersection = logical_and(pred, gt).sum()
union = logical_or(pred, gt).sum()
iou = intersection / union
```

这是单个图像的前景裂缝 IoU。结果表中的 `iou_mean` 是：

> per-image foreground crack IoU averaged over the test set

它不是常见语义分割论文里 background + foreground 两类平均的 class-averaged mIoU。

### 论文风险

稿件大量使用 `mIoU`，例如：

- CrackForest Text: 0.4355
- DeepCrack Text: 0.6658
- CrackForest Point: 0.1166
- DeepCrack Point: 0.3129

这些数值与 CSV 对得上，但 `mIoU` 这个标签可能被审稿人理解为 class-averaged mean IoU。由于背景类通常很大，如果包含背景类，mIoU 数值可能明显不同。

### 建议改法

建议在表格和正文中把 `mIoU` 改成以下之一：

- `Mean foreground IoU`
- `Crack IoU`
- `Foreground IoU`

如果保留 `mIoU`，Methods 中必须明确：

> We report per-image foreground crack IoU averaged over the test set, rather than class-averaged mIoU including background.

### Dice 与 F1 重复

当前二值像素级设置下，Dice 和 F1 使用同一组 TP/FP/FN，数值完全相同。表格同时报告 Dice 和 F1 会显得冗余。

建议主表保留：

- Mean foreground IoU
- Dice
- Precision
- Recall

F1 可放入 appendix 或删除。

## 3. GT-Assisted Alignment 与可部署性边界必须更醒目

### 问题

AMPF、ablation、mode combination 和 mean fusion 均使用 `oracle_gt_prompt_gt_alignment` 协议。代码中 alignment 通过 GT instance 与候选 mask 的 IoU 进行匹配：

- GT mask 分解为 connected components；
- text/box candidates 按 GT instance 选择 IoU 最大且超过阈值的候选；
- point candidates 按 prompt order 对齐并检查 IoU。

这意味着 AMPF/mean fusion 不是 fully automatic deployment method，而是 GT-assisted diagnostic analysis。

### 论文风险

摘要和结果中写：

> mean-fusion variant obtains the best overall mIoU

虽然正文后面有解释，但摘要中的 “best overall” 容易误导。审稿人可能认为作者把需要 GT alignment 的方法与 automatic text prompt 直接比较。

### 建议改法

#### 摘要中加限定

将：

> a simple mean-fusion variant obtains the best overall mIoU

改为：

> under the GT-assisted alignment protocol, a simple mean-fusion variant obtains the highest diagnostic foreground IoU

#### 表格中保留 protocol 列

所有主结果表建议保留或显式标注：

- `automatic_text_prompt`
- `oracle_gt_prompt`
- `oracle_gt_prompt_gt_alignment`
- `supervised_train_val_test`

不要只在 Methods 中说明一次。

#### 结论降调

当前结论可以保留，但建议改成：

> The practical deployable recommendation is SAM3 text prompting. The fusion results show that aligned multi-prompt information contains value, but the current alignment is GT-assisted and should be viewed as a diagnostic upper-bound analysis.

## 4. Box/Point 不应称为强 Oracle Upper Bound

### 问题

box prompt 是由整张 GT mask 的最小外接矩形生成，而不是每个裂缝实例一个 box，也不是人工优化 box。

point prompt 是每个 GT 连通域一个 centroid-derived point，而不是 skeleton point、多点 prompt 或交互式 foreground point。

### 论文风险

如果称为 “oracle upper bound”，审稿人会质疑：

- 为什么不是每个实例一个 box？
- 为什么 point 只用一个点？
- 为什么不保证点在前景？
- 为什么不评估多点 prompt？

### 建议改法

将 “oracle upper bound” 改成更保守的：

> GT-derived spatial prompt diagnostic protocol

或：

> GT-derived single-box and centroid-point protocols

不要泛化为 “box/point prompts underperform text prompts”。

更严谨的结论应为：

> Under the GT-derived single-box and centroid-point protocols used in this study, box and point prompting underperform automatic text prompting.

## 5. 结果路径和复现路径需要按提交根目录固定

### 问题

如果论文稿件放在 `sam3-demo/` 提交目录内，则 `manuscript.md` 中引用：

- `experiments/results/all_results.csv`
- `experiments/results/summary_stats.csv`
- `experiments/visualizations/...`

以 `sam3-demo/` 作为提交项目根目录时，上述路径是成立的，真实结果就在：

- `experiments/results/...`
- `experiments/visualizations/...`

外层工作区的 `paper_results/...` 可以作为投稿图表副本，但不应作为 `sam3-demo/` 项目内的唯一复现路径。

### 建议改法

将论文中的路径统一为 `experiments/results/...` 和 `experiments/visualizations/...`，用于 `sam3-demo/` 项目内复现。如果需要投稿材料副本，可另行说明外层 `paper_results/...` 是导出的 paper-ready copy。

推荐：

- 正文图表链接使用 `experiments/visualizations/...`；
- Data and Code Availability 中说明原始实验结果目录为 `experiments/results/...`；
- 不再引用外层工作区路径作为提交项目内的主路径。

## 6. Text Prompt 描述与实验记录有一处不一致

### 问题

代码中 text prompt 固定为：

```python
"crack"
```

但 `EXPERIMENTS.md` 中 protocol 表写：

> 使用固定文本 "crack, fracture, fissure, break"

这与 `experiments/prompt_generator.py` 的实际 text prompt 不一致。`fracture/fissure/break` 只出现在 stability score 的 synonym perturbation 中，不是 text single-mode baseline 的主 prompt。

### 建议改法

将 `EXPERIMENTS.md` 和论文中的描述统一为：

> Text mode uses the fixed prompt "crack". The synonyms "fracture", "fissure", and "break" are used only for text-prompt stability scoring in AMPF.

## 7. Supervised Baseline 描述与代码不完全一致

### 问题

`EXPERIMENTS.md` 中监督基线表写：

- U-Net: EfficientNet-b0 backbone
- DeepLabV3+: ResNet-50 backbone

但 `experiments/run_baselines.py` 实际代码中：

- U-Net 使用 `resnet34`
- DeepLabV3+ 使用 `resnet34`

此外，论文说 supervised baselines 是 reasonable reference models，这个定位是对的，但不应暗示它们代表充分调优的 SOTA。

### 建议改法

统一代码和论文描述。若以当前代码为准，应写：

> U-Net and DeepLabV3+ are implemented with segmentation_models_pytorch using ResNet-34 encoders and ImageNet initialization. They are intended as reference supervised baselines rather than exhaustively tuned SOTA crack segmentation models.

如果实际实验记录确实来自不同 backbone，则需要补充训练日志或重新生成结果，否则不要在论文中写 EfficientNet-b0 / ResNet-50。

## 8. 定性可视化证据不足

### 问题

当前表格结果完整，但论文缺少足够的真实预测 mask 可视化。已有 `run_experiments.py` 中 comparison figure 逻辑主要展示 IoU 文本或有限图像，不足以支撑：

- text 高召回低精度；
- point 为何失败；
- AMPF/mean fusion 在哪些情况下改善；
- fusion 如何引入 false negatives。

### 建议改法

新增 qualitative figure：

每行一个样本，每列为：

1. Original image
2. GT
3. Text
4. Box
5. Point
6. AMPF
7. Mean fusion

建议至少包含：

- 2 个 text 成功样本；
- 2 个 point 明显失败样本；
- 2 个 mean fusion 优于 text 的样本；
- 2 个 AMPF 或 fusion 退化样本。

可在图注中明确：

> Point prompts are centroid-derived and may be off-foreground or locally ambiguous for thin crack structures.

## 9. 需要补充的关键实验

### 高优先级

1. **Nearest-foreground point protocol**
   - 替换当前 centroid point；
   - 重新跑 point、point+text、box+point、AMPF、mean fusion；
   - 对比旧 centroid protocol 与新 foreground protocol。

2. **Per-instance box protocol**
   - 当前 box 是整张 GT mask 的一个 global box；
   - 新增每个 GT connected component 一个 box；
   - 判断 box prompting 是否确实弱，还是 global box 设计太粗。

3. **Automatic proposal protocol**
   - 用简单 image processing、edge/region proposal、YOLO proposal 或 human-in-the-loop box 模拟；
   - 用于替代 GT-assisted prompt/alignment；
   - 这是让 AMPF 从 diagnostic analysis 走向 deployable method 的关键。

### 中优先级

4. **Threshold sensitivity**
   - text confidence threshold 0.25；
   - fusion confidence threshold 0.5；
   - IoU matching threshold 0.1；
   - 建议至少做小范围 sensitivity analysis。

5. **Runtime analysis**
   - text single prompt；
   - point per connected component；
   - AMPF with stability perturbations；
   - 给出每张图平均耗时和显存。

6. **Statistical reporting**
   - 对关键 pairwise comparison 给出 mean delta、p-value、wins/losses；
   - 明确 p-values 是 descriptive support，不是跨协议公平优越性证明。

## 10. 建议的论文改写策略

### 标题建议

当前标题：

> Training-Free Crack Segmentation for Smart Infrastructure Inspection: A Systematic Evaluation of SAM3 Prompt Modes and Fusion Strategies

可以保留，但更稳的版本是：

> Protocol-Aware Evaluation of SAM3 Prompt Modes and Fusion Strategies for Training-Free Crack Segmentation

或者：

> How Much Does Multi-Prompt Fusion Help? A Protocol-Aware Evaluation of SAM3 for Crack Segmentation

### 摘要主线建议

摘要应突出三点：

1. text prompt 是唯一 fully automatic baseline；
2. box/point/AMPF 是 GT-derived or GT-assisted diagnostic protocols；
3. mean fusion 的最好结果只在 GT-assisted alignment 下成立。

### 贡献点建议

将贡献点改为：

1. Protocol-aware benchmark of SAM3 text, box, and point prompts for crack segmentation.
2. Explicit separation of automatic text prompting, GT-derived spatial prompting, and GT-assisted alignment.
3. Evidence that centroid-point and global-box protocols underperform text prompting for thin crack structures.
4. Diagnostic evidence that alignment is necessary for multi-prompt fusion, while confidence weighting is not yet well calibrated.
5. Reference comparison against supervised baselines under clearly stated implementation constraints.

## 11. 修订优先级清单

### 必须修

- 将 `mIoU` 改为 `Mean foreground IoU` 或明确其定义；
- 将 point prompt 表述改为 `centroid-derived point`；
- 去掉或弱化 `oracle upper bound`；
- 在摘要、结果表、结论中强化 GT-assisted alignment 限定；
- 若稿件不在 `sam3-demo/` 根目录内，则修正图表和结果路径；若以 `sam3-demo/` 为提交根目录，则保留 `experiments/...` 路径；
- 修正 text prompt 与 synonym perturbation 的描述；
- 修正监督 baseline backbone 描述。

### 强烈建议修

- 新增 nearest-foreground point protocol，并重跑 point 相关实验；
- 新增 per-instance box protocol；
- 增加真实 mask qualitative visualization；
- 将 Dice/F1 去重，主表保留 IoU/Dice/Precision/Recall。

### 可作为后续工作

- 自动 proposal + AMPF；
- 更大数据集；
- runtime/memory profiling；
- confidence calibration 或 learned fusion；
- crack-specific stability score。

## 12. 推荐写入 manuscript 的关键句

以下句子可直接改写进论文：

> We report mean foreground crack IoU, computed as per-image foreground IoU averaged over the test set, rather than class-averaged mIoU including background.

> The point protocol uses one centroid-derived positive point per GT connected component. Because centroids are not guaranteed to lie on crack pixels for concave or branching elongated structures, this protocol should be interpreted as a simple diagnostic point baseline rather than an oracle point upper bound.

> Box, point, AMPF, and mean-fusion variants use GT-derived prompts or GT-assisted alignment. They are therefore diagnostic protocols for understanding SAM3 behavior and prompt fusion, not fully automatic deployment methods.

> Under the GT-assisted alignment protocol, mean fusion achieves the highest diagnostic foreground IoU. The deployable practical baseline remains automatic text prompting.

> The supervised baselines are reference implementations intended to contextualize training-free SAM3 performance; they are not claimed to represent exhaustively tuned state-of-the-art crack segmentation systems.
