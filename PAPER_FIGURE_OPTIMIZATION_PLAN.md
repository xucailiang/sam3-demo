# Paper Figure and Evidence Optimization Plan

本文档用于指导 `manuscript_revised_protocol_aware.md` 后续优化实施，重点回应导师关于“缺少学术 figure、methodology 框架图、定量分布图、错误分析图、校准图、参数敏感性图”的建议。

核心目标不是简单增加图片数量，而是建立一条清晰的论文证据链：

1. 论文的方法论贡献是什么；
2. 不同 protocol 的 deployability 边界在哪里；
3. Text / AMPF / Mean Fusion 的性能差异是否稳定；
4. AMPF 是否真的只是改变 precision-recall trade-off；
5. 视觉失败模式是否支持正文讨论；
6. confidence weighting “不够校准”这个判断是否有足够实验支撑。

---

## 1. 推荐主文 Figure 结构

建议主文放 4 张核心图，Supplementary 放 2-3 张扩展图。

| Figure | 名称 | 主文/补充 | 当前可做性 | 目的 |
|---|---|---|---|---|
| Fig. 1 | Protocol Hierarchy & Deployability Framework | 主文 | 可直接做 | 解释论文核心方法论贡献 |
| Fig. 2 | Per-Image IoU Distribution Violin/Box Plot | 主文 | 可直接做 | 展示 per-image 稳定性和小样本不确定性 |
| Fig. 3 | Precision-Recall Operating Points Scatter | 主文 | 可直接做 | 支撑 AMPF 改变 PR trade-off 的结论 |
| Fig. 4 | Qualitative Failure Mode Analysis Grid | 主文 | 需要重新跑推理生成 masks | 支撑 over-segmentation、local recovery、false negatives 等错误分析 |
| Fig. S1 | Confidence Calibration Reliability Diagram | Supplementary 或后续主文 | 需要额外保存 candidate-level confidence | 支撑 confidence score 未校准的结论 |
| Fig. S2 | AMPF Hyperparameter Sensitivity Heatmap | Supplementary | 需要额外跑参数敏感性实验 | 证明 AMPF 权重选择是否鲁棒 |
| Fig. S3 | Inference Time vs Accuracy Scatter | Supplementary | 需要更细粒度 runtime profiling | 展示 accuracy-efficiency trade-off |

---

## 2. 可以直接做的优化项

以下内容可以基于现有 `experiments/results/all_results_v2.csv`、`summary_stats_v2.csv` 和论文现有 protocol 设计直接完成，不需要重新跑 SAM3 推理。

### 2.1 Fig. 1: Protocol Hierarchy & Deployability Framework Diagram

**可直接做。**

这是论文最重要的概念图，建议放在 Methodology 开头，作为 Figure 1。

建议图形结构：

```text
                 Deployable Protocol
                Text Prompt ("crack")
              no GT prompt, no GT alignment
                         |
                         v
          GT-Derived Diagnostic Protocol
                 Box Prompt / Point Prompt
              GT used only to generate prompts
                         |
                         v
        GT-Assisted Fusion Diagnostic Protocol
                AMPF / Mean Fusion / Ablations
        GT prompt generation + GT instance alignment
```

右侧增加应用场景：

| Protocol Level | Methods | Practical Meaning |
|---|---|---|
| Deployable | Text | similar visible surface-crack imagery: mobile inspection / UAV screening / low-label screening |
| GT-derived diagnostic | Box, Point | prompt-mode diagnosis / annotation workflow analysis |
| GT-assisted fusion diagnostic | AMPF, Mean Fusion | research diagnosis / fusion potential, not deployment |

建议视觉编码：

- 绿色：deployable；
- 黄色：GT-derived diagnostic；
- 红/橙：GT-assisted diagnostic；
- 箭头标注信息依赖逐步增加；
- 明确写出 “more GT assistance, less deployable”。

输出建议：

- `experiments/visualizations/fig_protocol_hierarchy.pdf`
- `experiments/visualizations/fig_protocol_hierarchy.png`

论文中应引用为：

> Figure 1 summarizes the protocol hierarchy used in this study. Only the text-prompt protocol is fully automatic; spatial prompts and fusion protocols use ground-truth-derived information and are therefore diagnostic.

---

### 2.2 Fig. 2: Per-Image IoU Distribution Violin/Box Plot

**可直接做。**

数据来源：

- `experiments/results/all_results_v2.csv`

推荐方法：

- Text
- Box
- Point
- AMPF
- Mean Fusion (`ablation_fusion_mean`)
- U-Net
- DeepLabV3+

图形结构：

- 左子图：CrackForest, n=24
- 右子图：DeepCrack, n=237
- 横轴：method
- 纵轴：per-image foreground IoU
- violin plot 或 box plot
- 叠加 jitter points 展示每张图的表现
- 标注关键 Wilcoxon 显著性：
  - Mean Fusion vs Text
  - AMPF vs Text
  - Text vs U-Net

注意：Mean Fusion / AMPF 与 Text 的显著性标注只能作为 protocol-aware descriptive comparison。图注必须说明 Mean Fusion / AMPF 是 GT-assisted diagnostic protocol，不能被解读为 fully deployable 方法优于 Text。

需要传达的信息：

1. CrackForest 测试集很小，分布解释要谨慎；
2. DeepCrack 更能支撑统计结论；
3. Text 的分布与 supervised baselines 接近；
4. Mean Fusion 的提升是 diagnostic，并非 deployable；
5. AMPF 相比 Text 没有稳定 IoU 提升。

输出建议：

- `experiments/visualizations/fig_iou_distribution.pdf`
- `experiments/visualizations/fig_iou_distribution.png`

实现脚本建议：

- 新增 `experiments/plot_iou_distribution.py`

---

### 2.3 Fig. 3: Precision-Recall Operating Points Scatter

**可直接做。**

数据来源：

- `experiments/results/summary_stats_v2.csv`
或从 `all_results_v2.csv` groupby 得到 mean precision / mean recall。

推荐方法：

- Text
- Box
- Point
- AMPF
- Mean Fusion
- U-Net
- DeepLabV3+
- YOLOv8-seg

图形结构：

- 横轴：Recall
- 纵轴：Precision
- 两个子图：CrackForest / DeepCrack
- 每个方法一个点；
- 用箭头连接 `Text -> AMPF -> Mean Fusion`；
- 可选：添加 F1-score contour lines。

注意：`Text -> AMPF -> Mean Fusion` 箭头表示 precision-recall operating point 的移动，不表示部署流程，也不表示后者在实际系统中自动可用。

需要传达的信息：

1. Text 是 high-recall / lower-precision baseline；
2. AMPF 从 Text 移动到 higher-precision / lower-recall 区域；
3. Mean Fusion 在 GT-assisted alignment 下获得更平衡的点；
4. YOLOv8-seg 对 thin crack masks 不占优。

输出建议：

- `experiments/visualizations/fig_precision_recall_operating_points.pdf`
- `experiments/visualizations/fig_precision_recall_operating_points.png`

实现脚本建议：

- 新增 `experiments/plot_precision_recall.py`

---

### 2.4 正文文字联动修改

**可直接做。**

添加以上 figure 后，正文需要同步微调：

1. Methodology 开头引用 Fig. 1；
2. Results 的 Single-Prompt / AMPF 部分引用 Fig. 2；
3. Results 或 Discussion 中 precision-recall trade-off 段落引用 Fig. 3；
4. Limitations 中保留可见表面裂缝 benchmark 的边界；
5. 避免写 “confidence is not calibrated” 这种强结论，除非后续完成 calibration figure。

---

## 3. 需要重新跑推理或额外实验后才能做的优化项

以下 figure 不能仅凭现有 summary CSV 可靠完成。若强行绘制，会有实验支撑不足或伪证据风险。

### 3.1 Fig. 4: Qualitative Failure Mode Analysis Grid

**需要重新跑推理生成 masks。**

当前状态：

- `experiments/visualizations/` 只有指标柱状图；
- `all_results_v2.csv` 保存的是 per-image metrics，不保存预测 mask；
- 已新增脚本：
  - `experiments/generate_qualitative_figure.py`
- 当前本机状态：
  - `models/sam3/sam3.pt` 存在；
  - `experiments/datasets/` 缺失；
  - 因此暂时无法实际生成 qualitative overlay。

建议图形结构：

| 行 | Failure / Success Mode | 目的 |
|---|---|---|
| 1 | Text Over-segmentation | 展示 high recall / low precision，阴影、纹理、接缝误分 |
| 2 | Point Local Recovery | 展示单点只恢复局部区域，漏掉长裂缝分支 |
| 3 | AMPF False Negative | 展示 confidence filtering 提高 precision 但压低 recall |
| 4 | Mean Fusion Complementary Recovery | 展示 GT-assisted alignment 下多 prompt 互补恢复 |

推荐列：

1. Original
2. Ground Truth
3. Text
4. Point
5. AMPF
6. Mean Fusion

注意事项：

- 必须使用真实模型输出；
- mask 使用半透明 overlay；
- FP 区域可用红色箭头；
- FN 区域可用黄色框；
- caption 中必须说明 Mean Fusion 是 GT-assisted diagnostic，不是 deployable。

实施前置条件：

1. 下载/恢复 `experiments/datasets/CrackForest-dataset`
2. 下载/恢复 `experiments/datasets/DeepCrack`
3. 运行：

```bash
.venv/bin/python experiments/generate_qualitative_figure.py \
  --datasets-dir experiments/datasets \
  --model models/sam3/sam3.pt
```

输出：

- `experiments/visualizations/fig_qualitative_prompt_examples.png`

后续增强：

- 先根据 `all_results_v2.csv` 找典型样本：
  - Text precision low / recall high 的样本；
  - Point IoU 远低于 Text 的样本；
  - AMPF recall 低于 Text 的样本；
  - Mean Fusion IoU 高于 Text 的样本；
- 再用这些 image index 作为 `--case dataset:index` 参数生成图。

---

### 3.2 Fig. S1: Confidence Calibration Reliability Diagram

**需要额外保存 candidate-level confidence 和 mask quality。**

导师建议很有价值，但现有结果文件不够。

当前缺失数据：

- 每个 SAM3 candidate 的 detection score；
- 每个 candidate 的 composite confidence；
- 每个 candidate 与 GT 的 IoU；
- 每个 confidence bin 中的实际质量；
- per-mode reliability curve。

不能直接从 `all_results_v2.csv` 推出 calibration，因为 CSV 只有最终 mask 的 per-image IoU / Dice / Precision / Recall。

建议新增实验记录字段：

| 字段 | 含义 |
|---|---|
| dataset | 数据集 |
| image_id | 图像 ID |
| method | text / box / point / ampf |
| candidate_id | SAM3 candidate index |
| prompt_mode | text / box / point |
| detection_score | SAM3 原始置信度 |
| stability_score | AMPF stability score |
| boundary_score | AMPF boundary score |
| composite_confidence | AMPF composite confidence |
| candidate_iou | candidate mask vs GT IoU |
| candidate_dice | candidate mask vs GT Dice |

Reliability diagram 建议：

- 横轴：mean predicted confidence；
- 纵轴：actual quality，可用 mean candidate IoU 或 fraction of candidates above IoU threshold；
- 对角线：perfect calibration；
- 曲线：Text / Box / Point / AMPF；
- 下方给 Expected Calibration Error (ECE)。

注意：

如果不做这个实验，正文中应避免强写：

> SAM3 confidence is not calibrated.

更稳的表述是：

> Ablation results suggest that the current confidence-weighted fusion rule is not reliably beneficial for crack segmentation.

---

### 3.3 Fig. S2: AMPF Hyperparameter Sensitivity Heatmap

**需要额外跑参数敏感性实验。**

当前 AMPF 权重：

```text
alpha = 0.40
beta  = 0.35
gamma = 0.25
```

导师建议：

- 固定 `gamma = 0.25`
- 变化 `alpha` 和 `beta`
- 约束 `alpha + beta + gamma = 1`

但严格来说，在固定 gamma 且权重和为 1 时，`beta = 0.75 - alpha`，不是独立二维网格。若要做二维 heatmap，需要允许 gamma 随 alpha/beta 变化：

```text
gamma = 1 - alpha - beta
alpha > 0, beta > 0, gamma > 0
```

推荐实验设计：

1. DeepCrack full set 或 50-image representative subset；
2. alpha ∈ {0.1, 0.2, ..., 0.8}
3. beta ∈ {0.1, 0.2, ..., 0.8}
4. gamma = 1 - alpha - beta，过滤 gamma <= 0 的组合；
5. 每组记录 mean foreground IoU / Dice / Precision / Recall。

输出：

- `experiments/results/ampf_weight_sensitivity.csv`
- `experiments/visualizations/fig_ampf_weight_sensitivity_heatmap.pdf`
- `experiments/visualizations/fig_ampf_weight_sensitivity_heatmap.png`

图形信息：

- heatmap 颜色：DeepCrack mean foreground IoU；
- 标注当前论文参数点 `(0.4, 0.35, 0.25)`；
- 可补一张 CrackForest 小热图，展示小样本不稳定性。

论文使用方式：

- 若结果显示大范围 plateau：说明 AMPF 对权重不敏感；
- 若当前点不是最优：进一步支持 “confidence weighting not calibrated / not optimized”；
- 若某些区域明显更好：可作为 future work 或更新 AMPF 配置。

---

### 3.4 Fig. S3: Inference Time vs Accuracy Scatter

**需要更细粒度 runtime profiling。**

当前 `EXPERIMENTS.md` 有阶段耗时，但不足以画严谨的 ms/image scatter：

- SAM3/AMPF CrackForest 约 12 min；
- DeepCrack full groups 耗时若干；
- supervised baselines 训练耗时；
- 但没有每种 method 的 inference time / image。

建议新增 profiling：

| method | dataset | n_images | total_time_s | mean_ms_per_image | std_ms_per_image | mean_iou |
|---|---|---:|---:|---:|---:|---:|

方法：

- Text
- Box
- Point
- AMPF
- Mean Fusion
- U-Net
- DeepLabV3+
- YOLOv8-seg

图形：

- 横轴：ms/image；
- 纵轴：mean foreground IoU；
- 点大小：memory or parameter count；
- 颜色：protocol type；
- 重点展示 Text 是否是 accuracy-efficiency sweet spot。

注意：

训练时间和推理时间要分开，不能混在同一张图里。

---

## 4. 推荐实施顺序

### Phase 1: 不跑模型，直接补主文图（已完成）

状态：已完成。已新增绘图脚本，生成 Fig. 1-3 的 PNG/PDF，并将三张图正式接入 `manuscript_revised_protocol_aware.md`。

1. [x] 新增 `experiments/plot_protocol_hierarchy.py`
2. [x] 新增 `experiments/plot_iou_distribution.py`
3. [x] 新增 `experiments/plot_precision_recall.py`
4. [x] 生成 Fig. 1-3 的 PNG/PDF
5. [x] 修改 `manuscript_revised_protocol_aware.md`，正式引用 Fig. 1-3

预计收益：

- 显著提高稿件学术呈现；
- 不增加实验风险；
- 直接回应导师“缺少学术 figure”和“methodology 框架图”的核心意见。

### Phase 2: 恢复数据集后生成 qualitative figure

优先级高，但依赖数据集。

1. 恢复 `experiments/datasets/`
2. 用 CSV 筛选典型样本；
3. 运行 `experiments/generate_qualitative_figure.py`
4. 生成 Fig. 4；
5. 将 `5.7 Qualitative Analysis` 从“应生成”改为正式结果描述。

预计收益：

- 支撑错误分析；
- 让审稿人看到模型真实行为；
- 提升应用类期刊接受度。

### Phase 3: 补充实验图

优先级中等，适合 supplementary。

1. Candidate-level confidence logging；
2. Reliability diagram + ECE；
3. AMPF weight sensitivity；
4. Inference-time profiling。

预计收益：

- 加强 “confidence weighting 不可靠” 的证据；
- 让 AMPF 分析更完整；
- 但需要额外计算和工程实现。

---

## 5. 当前稿件中应避免的强结论

在完成 Fig. S1 / Fig. S2 前，不建议写：

1. “SAM3 confidence is not calibrated.”
2. “AMPF weights are suboptimal.”
3. “Text prompt is generally deployable for all smart infrastructure crack inspection.”
4. “Mean Fusion outperforms supervised models.”
5. “The method is ready for operational deployment.”

建议使用更严谨表述：

1. “Ablation results suggest that the current confidence-weighted fusion rule is not reliably beneficial.”
2. “Within the evaluated visible surface-crack benchmarks, text prompting is a strong deployable baseline.”
3. “Mean Fusion provides a GT-assisted diagnostic positive result.”
4. “Broader validation is required before general infrastructure-level deployment claims.”

---

## 6. 最终主文图表建议排序

推荐主文结构：

1. **Figure 1**: Protocol hierarchy and deployability framework
2. **Table 1**: Protocol definitions
3. **Figure 2**: Per-image IoU distributions
4. **Table 2**: Main quantitative comparison
5. **Figure 3**: Precision-recall operating points
6. **Table 3**: Paired statistical comparisons
7. **Figure 4**: Qualitative failure mode analysis
8. **Table 4**: Ablation study

这样安排的逻辑是：

- 先解释 protocol；
- 再展示分布；
- 再展示均值；
- 再展示 PR trade-off；
- 再给统计检验；
- 最后用定性图解释失败模式。

---

## 7. 制图规范

1. 同时输出 PNG 和 PDF；
2. PDF 用于论文投稿，PNG 用于 Markdown 预览；
3. 使用 colorblind-friendly palette，例如 `viridis`, `tab10`, `ColorBrewer Set2`；
4. 避免纯红绿对比；
5. 所有图中统一方法颜色：
   - Text: blue
   - Box: orange
   - Point: purple
   - AMPF: red
   - Mean Fusion: green/teal
   - supervised baselines: gray/black
6. 所有图必须标注 dataset；
7. 对 GT-assisted 方法在图例中加 `diagnostic` 或 `GT-assisted`；
8. qualitative mask overlay 使用 alpha=0.4-0.5；
9. 图注中必须声明 Mean Fusion / AMPF 的 protocol caveat。

---

## 8. 立即可执行清单

### 可立即实施

- [x] 画 Fig. 1 protocol hierarchy；
- [x] 画 Fig. 2 per-image IoU distribution；
- [x] 画 Fig. 3 precision-recall scatter；
- [x] 正文引用 Fig. 1-3；
- [x] 所有涉及 AMPF / Mean Fusion 的图注显式标注 `GT-assisted diagnostic`；
- [x] 保留并完善 visible surface-crack benchmark 限定；
- [x] 删除或弱化所有超出数据集范围的泛化结论。

### 需要数据集和重新推理

- [ ] 生成 Fig. 4 qualitative failure mode grid；
- [ ] 选择典型 failure/success cases；
- [ ] 保存 overlay masks；
- [ ] 将 qualitative section 从计划性描述改成结果性描述。

### 需要新增实验设计

- [ ] Candidate-level confidence logging；
- [ ] Reliability diagram and ECE；
- [ ] AMPF hyperparameter sensitivity；
- [ ] runtime profiling；
- [ ] accuracy-time scatter。
