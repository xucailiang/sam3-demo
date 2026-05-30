# Nature-Reader Style Review of `manuscript_revised_protocol_aware.md`

审阅日期: 2026-05-28  
审阅对象:

- Markdown 源稿: `manuscript_revised_protocol_aware.md`
- PDF 排版稿: `Protocol-Aware_Evaluation_of_SAM3_Crack_Segmentation.pdf`
- 关联证据: `experiments/results/*`, `experiments/visualizations/*`, `experiments/*.py`

## 总体判断

这版稿件已经把论文主线收束到正确方向: protocol-aware evaluation，而不是把 AMPF 包装成可部署优胜方法。当前最有说服力的贡献是:

> 在 CrackForest 和 DeepCrack 这两个可见表面裂缝基准上，SAM3 固定文本提示 `"crack"` 是唯一 fully automatic 的 training-free SAM3 协议；box、point、AMPF 和 mean fusion 是 GT-derived 或 GT-assisted diagnostic protocols，不能直接作为 deployable method 宣称。

主要数值和结果文件基本一致，旧版 review 中的若干硬伤已经修正: point protocol 已改为 nearest-foreground point，IoU 已明确为 mean foreground IoU，baseline epochs 已写成 100，refs 20/21 也已补全。当前最大投稿风险转移到 PDF 排版、图注编号、补充图处理、统计/补充实验描述深度、以及最终投稿声明完整性。

不建议直接提交当前 PDF。建议先修 PDF 图注/编号问题，再补一轮方法和可复现性细节。

## 主要问题

### 1. PDF 图注被重复编号，补充图编号与正文不一致

**严重级别: 高。**

Markdown 中每张图采用了:

```markdown
![Figure 1. ...](...)

**Figure 1.** ...
```

这在 Markdown 阅读中可接受，但 pandoc/LaTeX 生成 PDF 时会把图片 alt text 自动转成浮动体 caption，导致 PDF 中出现重复图注。

PDF 抽取结果显示:

- p.4: `Figure 1: Figure 1. Protocol hierarchy and deployability framework.`，随后又出现一遍 `Figure 1. Protocol hierarchy...`
- p.8: `Figure 2: Figure 2. Per-image foreground IoU distributions.`
- p.9: `Figure 3: Figure 3. Precision-recall operating points.`
- p.12: `Figure 4: Figure 4. Qualitative failure mode analysis...`
- p.13: `Figure 5: Figure S1. Confidence calibration reliability diagram.`
- p.14: `Figure 6: Figure S2. AMPF weight sensitivity heatmap.`
- p.15: `Figure 7: Figure S3. Runtime and accuracy profiling.`

这会造成两个问题:

1. 正文主图图注重复，视觉上像排版错误。
2. Supplementary figures 在 PDF 中被自动编号成 Figure 5, Figure 6, Figure 7，但正文又叫 Figure S1-S3，交叉引用会混乱。

更麻烦的是 PDF 文本顺序显示 p.13 已出现 Figure S2 的文字图注，但 Figure S2 图像本体在 p.14 才出现，说明浮动体把图与说明拆开了。对于投稿稿件，这属于必须修的排版问题。

建议:

- 对主图: 让图片 alt text 简短或空，不让 pandoc 自动生成完整 caption；只保留正式 caption 一处。
- 对补充图: 如果目标期刊允许 supplementary figures 放正文末尾，应使用 LaTeX/Pandoc 模板控制编号为 `Figure S1`、`Figure S2`、`Figure S3`，或明确改为 main Figure 5-7，不要混用。
- 避免图像和图注跨页分离，尤其是 Figure S2。

### 2. `Author Contributions` 仍是占位符

**严重级别: 高。**

Markdown 第 406-408 行仍为:

```markdown
## Author Contributions

[To be completed according to the final author list.]
```

PDF p.19 也会呈现这个占位文本。投稿前必须替换为真实作者贡献。即便期刊投稿系统另有 CRediT 表单，稿件正文里也不应保留 bracketed placeholder。

建议至少改成期刊接受的占位声明，例如投稿前版本可写:

> Author contributions will be finalized according to the confirmed author list before submission.

但真正提交版本应使用 CRediT taxonomy 或目标期刊格式。

### 3. Supplementary diagnostics 混入主文后，命名和叙事边界不够清楚

**严重级别: 中高。**

第 5.8 节将 Figure S1-S3 放在 Results 主文里，但 PDF 实际把它们自动编号为 Figure 5-7。内容上这些分析是 supplementary diagnostics，叙事上却进入主结果。审稿人可能会问: 这些是 main results 还是 supplement?

当前文字已经说 "diagnostic and subset-specific"，这很好。但排版和编号没有配合这个定位。

建议二选一:

- 如果保留在主文: 改名为 Figure 5-7，并删除 S-prefix；标题可叫 "Additional diagnostics" 而不是 supplementary。
- 如果作为补充材料: 从主文正文移出完整图，只保留一句 summarizing reference，并在 supplementary file 中编号 Figure S1-S3。

目前混用会降低稿件成熟度。

### 4. Table 5.3 和 5.4 的加粗仍可能制造跨协议误读

**严重级别: 中。**

稿件已经在第 261-274 行对 Table 5.6 增加了 protocol-stratified 解释，这比旧版好很多。但第 211-217 行和第 231-239 行仍把 automatic text、GT-assisted AMPF、GT-assisted mean fusion放在一个表内并加粗最佳值。

问题不在数值，而在视觉语义:

- 第 213 行 Text 是 deployable baseline。
- 第 214-217 行是 GT-assisted fusion diagnostics。
- 第 239 行 mean fusion 在两个数据集都被加粗，但它依赖 GT-assisted alignment。

虽然正文第 243 行说明 "under the GT-assisted alignment protocol"，但表格第一眼仍像 leaderboard。

建议:

- 表 5.3 和 5.4 的加粗只在同一 protocol family 内使用，或干脆取消加粗。
- 在表题或表后第一句明确: "Bold values indicate within-table numeric maxima only and are not deployability comparisons."
- 更稳的做法是把 automatic、GT-derived、GT-assisted 三类结果拆成三个小表。

### 5. `S_stab` 的可复现细节仍不足

**严重级别: 中。**

第 124 行写:

> text prompts are varied using synonyms, boxes are shifted slightly, and points are perturbed by a few pixels.

第 318 行又讨论 `"fracture"` 和 `"fissure"` 可能改变语义。但 Methods 中没有列出完整 synonym list、box shift 幅度、point perturbation 半径、perturbation 次数、随机种子处理方式。

因为论文的一个结论是 confidence-weighted AMPF 不可靠，而 `S_stab` 是 AMPF 的核心构件之一，审稿人需要知道这个 stability score 到底如何生成。

建议在 Methods 或 appendix 中补充:

- text synonym list;
- box perturbation offsets or percentage;
- point perturbation radius and samples;
- whether perturbations are deterministic under seed 42;
- whether stability score uses IoU consistency, mask area consistency, or another definition。

否则 `S_stab` 的负面结论可解释性不足。

### 6. Calibration ECE 的定义没有进入稿件正文

**严重级别: 中。**

Figure S1 caption 第 290 行报告:

- text ECE = 0.287
- box ECE = 0.362
- point ECE = 0.469

代码 `experiments/plot_calibration.py` 中 ECE 是 10 bins 下 `|mean candidate IoU - mean composite confidence|` 的加权平均。这个不是分类校准里常见的 accuracy-confidence ECE，而是 candidate-IoU calibration error。

建议在稿件中明确:

> Here ECE is computed over candidate masks by binning composite confidence and comparing it with mean candidate IoU, rather than binary classification accuracy.

否则读者可能误解 ECE 指标含义。

### 7. Runtime profiling 的实验说明仍偏薄

**严重级别: 中。**

第 298 行报告了 20-image profiling subset 的 mean ms/image。CSV 中有 std:

- point: 2990.8 ± 4570.9 ms/image
- AMPF: 8866.2 ± 9097.7 ms/image
- mean fusion: 3248.0 ± 4693.0 ms/image

标准差很大，说明均值受图像中 connected components 或候选数量强烈影响。第 348 行已经承认 runtime 只基于 20 张图，但 Figure S3 caption 最好也给出 variability 或说明为什么只报均值。

另外，`experiments/run_profiling.py` 使用 `time.perf_counter()` 包住异步 segmentation calls，但稿件没有说明是否做 GPU synchronization、是否包含 mask decoding、是否包含 prompt generation、是否包含 model warmup。脚本确实有 warmup=2 和 seed=42，但正文只说 after warmup。

建议:

- Caption 加 `mean ± std ms/image`，或补 median/IQR。
- 明确 runtime includes segmentation call and mask decoding under this implementation。
- 如果使用 CUDA，说明是否同步；若未同步，则把 runtime 定位为 implementation-level wall-clock profiling。

### 8. Dataset 获取和许可信息仍不够正式

**严重级别: 中。**

第 366 行写 CrackForest 和 DeepCrack should be obtained from original providers under licenses。这个方向正确，但正式投稿通常需要更具体:

- 数据集 URL 或 DOI；
- split 文件或 split generation seed 是否随包提供；
- CrackForest 的 MATLAB label mapping 是否有脚本可复现；
- DeepCrack official test split 的来源；
- 是否提供 trained supervised checkpoints。

当前 Data and Code Availability 更像内部复现说明，而不是正式 journal statement。

建议把 availability 拆成四句:

1. Public datasets and access links/licenses.
2. Code and result CSV repository/supplement URL.
3. Model checkpoint acquisition and redistribution constraints.
4. What is and is not included: per-image metrics included, full predicted masks/checkpoints not included unless actually supplied。

### 9. 引用组 `[14-18]` 仍略混

**严重级别: 低到中。**

第 5 行和第 17 行把 promptable vision foundation models / semantic or spatial prompts 引到 `[14-18]`。其中 [18] 是 Grounding DINO，主要是 open-set object detection / grounding，不是 segmentation model 本身。第 41 行写 SAM3 text/exemplar concept segmentation 引 `[16,18]` 也容易混，因为 [18] 不是 SAM3。

建议:

- Segment Anything family / promptable segmentation: `[14-17]`
- open-vocabulary grounding or grounding-assisted segmentation context: `[18,22]`

这不是硬伤，但会让 related work 更干净。

### 10. 标题仍偏长，且 "Fusion Strategies" 可能放大 AMPF 期待

**严重级别: 低到中。**

当前标题:

> Protocol-Aware Evaluation of SAM3 Prompt Modes and Fusion Strategies for Training-Free Crack Segmentation in Smart Infrastructure Inspection

方向正确，但仍很长，并且 "Fusion Strategies" 在标题中占比较高。由于主结论是 AMPF 不显著改善 IoU、mean fusion 只是 GT-assisted diagnostic result，标题可能让审稿人期待一个更强的 fusion 方法贡献。

建议压缩为:

> Protocol-Aware Evaluation of SAM3 for Training-Free Crack Segmentation

或:

> Protocol-Aware Evaluation of SAM3 Prompting for Training-Free Crack Segmentation

副标题或摘要中再交代 prompt modes and fusion diagnostics。

## 已修正或基本通过的部分

### Point protocol 与旧版 review 相比已明显改善

Markdown 第 61、98、198-205、342 行都已明确 nearest-foreground point protocol。代码 `experiments/prompt_generator.py` 也确实实现了 centroid fallback 到同一 connected component 中最近 foreground pixel。测试文件中已有 U-shape、C-ring、multi-component 和 nearest-pixel regression tests。

这部分现在基本过关。后续只需避免把 single-point protocol 写成 oracle upper bound。

### Mean foreground IoU 定义已清楚

第 55 和第 149 行明确 IoU 是 per-image foreground crack IoU averaged over test set，不是 class-averaged mIoU。主表也写 IoU 而不是 mIoU。这个旧问题已修正。

### Baseline epochs 与结果文件基本一致

第 172 行写 U-Net/DeepLabV3+ 和 YOLOv8-seg 均训练 100 epochs。当前 `summary_stats_v2.csv` 中监督结果与正文表格数值一致。旧版 50/100 epochs 不一致的问题已基本消除。

### 主要结果数值与 CSV 对得上

抽查 `experiments/results/summary_stats_v2.csv`:

- Text CrackForest IoU/Dice = 0.435491/0.599980，对应正文 0.4355/0.6000。
- Text DeepCrack IoU/Dice = 0.665818/0.786627，对应正文 0.6658/0.7866。
- Point DeepCrack IoU = 0.519930，对应正文 0.5199。
- AMPF DeepCrack IoU = 0.667089，对应正文 0.6671。
- Mean fusion DeepCrack IoU = 0.681529，对应正文 0.6815。
- U-Net DeepCrack IoU = 0.666140，对应正文 0.6661。

未发现核心表格数值转录错误。

### 图像资产存在

Markdown 引用的主要 PNG 图像均存在:

- `experiments/visualizations/fig_protocol_hierarchy.png`
- `experiments/visualizations/fig_iou_distribution.png`
- `experiments/visualizations/fig_precision_recall_operating_points.png`
- `experiments/visualizations/fig_qualitative_failure_modes.png`
- `experiments/visualizations/fig_calibration_reliability.png`
- `experiments/visualizations/fig_ampf_weight_sensitivity_heatmap.png`
- `experiments/visualizations/fig_runtime_accuracy_scatter.png`

问题不在文件缺失，而在 PDF 浮动体和图注生成方式。

## 建议修订优先级

1. 修 PDF 图注: 去掉重复 caption，统一 main/supplement figure numbering，避免 Figure S2 图注与图像跨页。
2. 替换 `Author Contributions` 占位符。
3. 决定 Figure S1-S3 是主文 Figure 5-7 还是 supplementary file，不要混用。
4. 给 `S_stab`、calibration ECE、runtime profiling 补充方法细节。
5. 弱化表格加粗，避免 GT-assisted mean fusion 被视觉呈现为 deployable winner。
6. 正式化 Data and Code Availability。
7. 精简引用组 `[14-18]` 和标题。

## 可投稿性判断

从科学叙事看，这版已经具备投稿基础: 贡献边界清楚，负结果也能自洽，核心数值与结果文件一致。当前阻碍主要是 manuscript-to-PDF production quality 和 reproducibility details。

修完上述高优先级问题后，建议投稿定位保持为:

> empirical protocol audit / training-free baseline study / diagnostic fusion analysis

不要把 AMPF 或 mean fusion 写成可部署算法贡献。最稳的主张仍是: SAM3 text prompt is a strong deployable baseline under the evaluated benchmark conditions; multi-prompt fusion remains diagnostic until GT-assisted prompting and alignment are replaced by automatic or human-in-the-loop mechanisms.
