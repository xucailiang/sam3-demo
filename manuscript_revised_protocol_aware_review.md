# Review of `manuscript_revised_protocol_aware.md`

审阅日期: 2026-05-28

## 总体判断

这版手稿的核心定位已经比早期版本稳健很多: 它不是强行宣称 AMPF 优于单提示或监督模型, 而是把贡献收束为 protocol-aware evaluation。这个方向是成立的, 也更容易经得住审稿。最强卖点应继续保持为:

> 在 CrackForest 和 DeepCrack 这两个可见表面裂缝基准上, SAM3 固定文本提示 `"crack"` 是当前唯一完全自动、可部署的 training-free SAM3 协议; GT-derived spatial prompts 和 GT-assisted fusion 是诊断性实验, 不能被包装成可部署算法性能。

不建议直接投稿。主要问题不是论文故事线, 而是复现实验设置、统计解释、参考文献完整性和部分术语边界仍有可被审稿人质疑的缺口。

## 主要问题

### 1. Supervised baseline 的训练轮数与代码不一致

原稿在 Implementation Environment 写道 U-Net 和 DeepLabV3+ 训练 50 epochs, YOLOv8-seg 训练 100 epochs: `manuscript_revised_protocol_aware.md:172`。

但当前 `experiments/run_supervised.py` 的实际调用是:

- 文件头注释写 50 epochs: `experiments/run_supervised.py:4`
- 主函数实际传入 `epochs=100`: `experiments/run_supervised.py:57`

这是硬伤。审稿人或复现者会认为 baseline 设置不可追溯。建议二选一:

- 如果最终表格结果来自 100 epochs, 把手稿改为 "trained for 100 epochs", 并同步更新脚本头部注释。
- 如果作者希望论文声明 50 epochs, 则重新运行 50 epochs baselines, 并更新 `summary_stats_v2.csv`, `paper_tables_v2.tex` 和正文数值。

当前表中 supervised 结果与 `summary_stats_v2.csv` 对得上, 例如 U-Net DeepCrack IoU = 0.6661: `experiments/results/summary_stats_v2.csv:31`。问题不是数值转录, 而是实验设置描述不可靠。

### 2. 统计显著性缺少多重比较和效应量控制

Section 5.5 报告了 9 个 Wilcoxon paired comparisons: `manuscript_revised_protocol_aware.md:245-257`。目前只说这些是 descriptive non-parametric paired tests: `manuscript_revised_protocol_aware.md:153`, 这有帮助, 但还不够。

风险点:

- 同一批 per-image IoU 被重复用于多组比较, 没有说明是否进行 Holm/Benjamini-Hochberg/FDR 修正。
- 只给 p-value, 没有给 effect size 或 confidence interval。DeepCrack Mean fusion - Text 虽然 p=0.00149, 但 mean difference 只有 +0.0157: `manuscript_revised_protocol_aware.md:251`。
- CrackForest n=24, p=0.00225 的 mean fusion gain 仍在 GT-assisted protocol 内, 容易被误读为强泛化证据。

建议在 Methods 或 Section 5.5 加一句:

> Because multiple paired comparisons are reported, p-values are treated as descriptive unless explicitly adjusted; the practical interpretation is based on protocol type, mean/median paired differences, and effect size rather than p-values alone.

更强的做法是补充 Holm-adjusted p-values, Cliff's delta 或 rank-biserial correlation。至少应在表头或脚注中标明 "unadjusted Wilcoxon p-value"。

### 3. GT-assisted fusion 与 supervised baseline 放在同一表时仍有误读风险

原稿已经在表格里保留 Protocol 列, 这是正确的: `manuscript_revised_protocol_aware.md:263-270`。但表 5.6 仍把 Text, AMPF, Mean fusion, U-Net, DeepLabV3+, YOLOv8-seg 放在一个 leaderboard 里, 并对 Mean fusion 的 CrackForest/DeepCrack IoU 加粗: `manuscript_revised_protocol_aware.md:267`。

问题是 Mean fusion 是 `oracle_gt_prompt_gt_alignment`, 与 deployable Text 和 supervised baselines 不在同一可部署层级。虽然正文下一句解释了不可直接比较: `manuscript_revised_protocol_aware.md:272`, 但视觉上还是像一个总榜第一。

建议:

- 表 5.6 改名为 "Protocol-stratified comparison with supervised baselines"。
- 只在每个 protocol stratum 内加粗, 或不要对 GT-assisted Mean fusion 进行全表加粗。
- 在 caption 中明确: "Bold values are not intended to indicate deployable superiority across protocols."

### 4. AMPF 的方法贡献定位还可以再降一点

当前 Introduction 第 3 条贡献写 "evaluates an Adaptive Multi-Prompt Fusion pipeline": `manuscript_revised_protocol_aware.md:25`, 这是安全的。但标题仍包含 "Fusion Strategies", Abstract 也较早引入 AMPF: `manuscript_revised_protocol_aware.md:7`。如果目标期刊偏应用或工程, 审稿人会追问: 既然 AMPF 不显著提升 IoU, 为什么要作为主要方法贡献?

建议把 AMPF 定位从 "pipeline contribution" 再改成 "diagnostic fusion probe":

> It uses an AMPF-style diagnostic fusion pipeline to test whether aligned multi-prompt candidates contain complementary signal under known instance correspondence.

这样即使 AMPF 被证明不优, 论文逻辑也不会被削弱; 负结果本身成为协议审计的一部分。

### 5. Runtime profiling 子集太小, 且只报告均值

Figure S3 的 runtime 只基于 20 张 DeepCrack profiling subset: `manuscript_revised_protocol_aware.md:296`, 对应文件也显示 `n_images=20`: `experiments/results/runtime_profiling.csv:2-6`。

当前讨论说 "text prompting is much cheaper than AMPF" 是成立的, 因为差距很大。但如果作为补充诊断, 建议加入:

- median ms/image 或 interquartile range, 因为 point/AMPF 的 std 很大。
- 明确是否包含 image loading、model warmup、GPU synchronization。
- 说明 subset selection 是否固定 seed, 是否与 qualitative examples 重叠。

这不一定要做成主实验, 但至少要避免让 runtime 结果看起来比实际更全面。

### 6. Figure 4 的代表性选择需要更可复现

Figure 4 的 caption 给出四个 DeepCrack image ids 和解释: `manuscript_revised_protocol_aware.md:278`。正文说 case selection script 是 `experiments/find_representative_cases.py`: `manuscript_revised_protocol_aware.md:280`。

建议在 Methods 或 Supplementary caption 中说明四行样例的选择规则, 例如:

- text over-segmentation: high recall / low precision case
- point local failure: large Text-Point IoU gap
- AMPF false negative: largest AMPF-Text IoU drop
- mean fusion recovery: largest Mean-Text IoU gain

否则审稿人可能认为 qualitative examples 是 cherry-picked。已有脚本名不够, 需要一句选择标准。

### 7. 参考文献 20 和 21 信息过粗

参考文献 20 现在写成 "Owor, N.J.; et al. PaveSAM..." `manuscript_revised_protocol_aware.md:433`。arXiv 页面显示完整作者为 Neema Jakisa Owor, Yaw Adu-Gyamfi, Armstrong Aboah, Mark Amo-Boateng, 题名和 2024 arXiv 信息可核对。建议补全作者和 arXiv ID `arXiv:2409.07295`。

参考文献 21 现在写成 "Zhang, Y.; et al. EVF-SAM..." `manuscript_revised_protocol_aware.md:434`。arXiv 页面显示 EVF-SAM 为 `arXiv:2406.20076`, v5 revised 2025。建议补全至少首三作者和 arXiv ID, 并把年份写成 2024 或 "arXiv:2406.20076, 2024; revised 2025"。

参考文献 16 的 SAM3 条目基本正确, 但 arXiv 当前版本是 v2, last revised 28 Mar 2026。若投稿时引用最新版, 建议写 `arXiv:2511.16719, 2025; revised 2026` 或直接按目标期刊格式引用 arXiv v2。

### 8. Related Work 中 [14-18] 的引用范围略混

Introduction 写 "Promptable vision foundation models ... queried using semantic or spatial prompts [14-18]": `manuscript_revised_protocol_aware.md:5`。这组引用里 [18] 是 Grounding DINO, 它更接近 open-set detection/grounding, 不是 segmentation model 本身; [22] Grounded SAM 更接近 "text grounding + SAM segmentation" 的组合。

建议:

- 对 "Segment Anything family / promptable segmentation" 使用 [14-17]。
- 对 "open-vocabulary grounding that can be coupled with segmentation" 使用 [18,22]。

这样相关工作的引用链更精确。

### 9. Data and Code Availability 仍不是正式投稿格式

Data and Code Availability 当前列出了本地结果和脚本路径: `manuscript_revised_protocol_aware.md:362-402`。这对内部复现很好, 但正式投稿还需要更清楚地区分:

- public datasets 的获取链接和许可;
- SAM3 weights 是否可再分发, 若不可, 如何申请或下载;
- repo/supplementary package 的公开 URL 或匿名审稿包;
- result CSV 是否包含所有 per-image predictions, 还是只包含 metrics;
- 是否包含 trained supervised checkpoints。

当前最后一句已经提到 SAM3 weights redistribution: `manuscript_revised_protocol_aware.md:402`, 但建议前置到 Availability 段落开头, 因为这是复现门槛。

### 10. Author Contributions / Conflicts of Interest 仍是占位文本

`manuscript_revised_protocol_aware.md:404-410` 仍包含 "should be completed" 和 "should declare"。如果这是投稿前稿件, 必须替换为真实声明。占位文本会直接暴露为未完成稿。

## 次要修改建议

- 标题略长。可改为: "Protocol-Aware Evaluation of SAM3 for Training-Free Crack Segmentation"。副标题或摘要中再放 "prompt modes and fusion strategies"。
- Abstract 信息密度高但可读性偏硬。建议拆分最后两句, 先给 deployable finding, 再给 diagnostic fusion finding。
- `asset priors` 在 `manuscript_revised_protocol_aware.md:80` 稍显突兀, 建议改为 `infrastructure-specific priors` 或 `asset-location priors`。
- Methods 中 `Text mode uses a SAM3 confidence threshold of 0.25`: `manuscript_revised_protocol_aware.md:106`。需要说明该阈值来源: default, pilot tuning, or fixed before evaluation。
- AMPF 的 `S_stab` prompt perturbation 使用 synonyms: `manuscript_revised_protocol_aware.md:124`。建议列出具体 synonyms 或放补充材料, 因为 Discussion 说 synonyms 可能改变语义: `manuscript_revised_protocol_aware.md:316`。
- Figure S2 中当前 paper setting 不在网格点上: `manuscript_revised_protocol_aware.md:292`。这很好, 但建议解释为什么原设置没有直接包含在 sensitivity grid 中。
- Dice 和 F1 在二值像素设置下相同, Methods 已说明: `manuscript_revised_protocol_aware.md:149`。主表最好只保留 Dice, F1 放补充或 CSV。
- Table 5.1 中 "automatic text prompt" 与 protocol 表里的 `automatic_text_prompt` 命名不完全一致: `manuscript_revised_protocol_aware.md:182`。建议全稿统一 snake_case 或 human-readable label, 不要混用。

## 优先级修订清单

1. 先修 supervised baseline epochs 的代码/手稿不一致。
2. 给 Section 5.5 加多重比较/未校正 p-value/效应量说明。
3. 调整 Table 5.6 的加粗和 caption, 避免 GT-assisted mean fusion 被视觉呈现为 deployable winner。
4. 补全 refs 20/21, 更新 SAM3 arXiv v2 信息。
5. 替换 Author Contributions 和 Conflicts of Interest 占位文本。
6. 给 Figure 4 case selection 和 runtime subset selection 增加可复现说明。
7. 在 Data and Code Availability 中加入 dataset/license/checkpoint/repo/supplementary package 的正式说明。

## 可投稿性判断

完成上述修订后, 这篇稿子的可信度会明显提高。现有结果文件与正文主要数值基本一致, 图像文件也齐全; 当前最大风险是复现实验描述和统计解释, 不是实验结果本身。

建议投稿定位保持保守: empirical evaluation / protocol audit / training-free baseline study。不要把 AMPF 包装成主要可部署方法, 也不要把 Mean fusion 写成跨协议 winner。
