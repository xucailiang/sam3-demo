# Paper Figure and Evidence Optimization Plan — 实施完成报告

本文档记录 `manuscript_revised_protocol_aware.md` 全部图表的实施过程和实验结果。所有 Phase 1-3 任务均已完成 (2026-05-28)。

核心论文证据链：

1. 论文的方法论贡献 → Fig. 1
2. 不同 protocol 的 deployability 边界 → Fig. 1 + Table 1
3. Text / AMPF / Mean Fusion 的性能差异是否稳定 → Fig. 2 + Fig. 3
4. AMPF 是否真的只是改变 precision-recall trade-off → Fig. 3 + Fig. S2
5. 视觉失败模式是否支持正文讨论 → Fig. 4
6. confidence weighting "不够校准"的判断是否有足够实验支撑 → Fig. S1 + Fig. S2

---

## 1. 最终图表清单

| Figure | 名称 | 位置 | 状态 | 关键发现 |
|---|---|---|---|---|
| Fig. 1 | Protocol Hierarchy & Deployability Framework | 主文 Methodology | ✅ | 三级协议体系 |
| Fig. 2 | Per-Image IoU Distribution | 主文 Results | ✅ | Text 分布与 U-Net 接近 |
| Fig. 3 | Precision-Recall Operating Points | 主文 Results | ✅ | AMPF 向高 precision/低 recall 移动 |
| Fig. 4 | Qualitative Failure Mode Analysis | 主文 Results | ✅ | 4 类典型故障模式 |
| Fig. S1 | Confidence Calibration Reliability Diagram | Supplementary | ✅ | DeepCrack candidate-level ECE 0.29-0.47，置信度显著高估 IoU |
| Fig. S2 | AMPF Weight Sensitivity Heatmap | Supplementary | ✅ | 30-image DeepCrack subset 中 IoU 变异系数仅 0.82%，呈 broad plateau |
| Fig. S3 | Inference Time vs Accuracy Scatter | Supplementary | ✅ | 20-image DeepCrack profiling subset: Text 84ms/img, AMPF 8866ms/img |

---

## 2. Phase 1: 主文 Fig. 1-3（已完成）

基于 `all_results_v2.csv` 直接绘制，不需要额外推理。

### Fig. 1: Protocol Hierarchy & Deployability Framework

- 脚本: `experiments/plot_protocol_hierarchy.py`
- 输出: `experiments/visualizations/fig_protocol_hierarchy.{png,pdf}`
- 内容: 三级协议层次（Deployable → GT-derived diagnostic → GT-assisted fusion diagnostic），右侧应用场景表
- 论文引用: Methodology 开头，Figure 1

### Fig. 2: Per-Image IoU Distribution

- 脚本: `experiments/plot_iou_distribution.py`
- 输出: `experiments/visualizations/fig_iou_distribution.{png,pdf}`
- 内容: CrackForest (n=24) / DeepCrack (n=237) 双面板 violin/box plot，叠加 jitter points
- 方法: Text, Box, Point, AMPF, Mean Fusion, U-Net, DeepLabV3+
- 标注关键 Wilcoxon 显著性

### Fig. 3: Precision-Recall Operating Points

- 脚本: `experiments/plot_precision_recall.py`
- 输出: `experiments/visualizations/fig_precision_recall_operating_points.{png,pdf}`
- 内容: 双面板 scatter，箭头标注 Text → AMPF → Mean Fusion 的 PR 移动
- 方法: Text, Box, Point, AMPF, Mean Fusion, U-Net, DeepLabV3+, YOLOv8-seg

### 正文联动

- Methodology 引用 Fig. 1
- Section 5.1 引用 Fig. 2
- Section 5.3 引用 Fig. 3
- 所有图注显式标注 GT-assisted diagnostic

---

## 3. Phase 2: Fig. 4 Qualitative Failure Mode Analysis（已完成）

### 3.1 实施步骤

**Step 1 — 筛选典型案例**

- 脚本: `experiments/find_representative_cases.py`
- 数据源: `experiments/results/all_results_v2.csv` (4176 per-image rows; 4177 CSV lines including header)
- 方法: pivot table + 排名，找出每种 failure mode 的最佳样本

**Step 2 — 生成 qualitative overlay**

- 脚本: `experiments/generate_qualitative_figure.py`
- 模型: `models/sam3/sam3.pt`
- 数据集: `experiments/datasets/DeepCrack/` (test split)
- 命令:

```bash
.venv/bin/python experiments/generate_qualitative_figure.py \
  --datasets-dir experiments/datasets \
  --model models/sam3/sam3.pt \
  --case deepcrack:168 \
  --case deepcrack:86 \
  --case deepcrack:17 \
  --case deepcrack:178 \
  --output experiments/visualizations/fig_qualitative_failure_modes.png
```

**Step 3 — 更新论文**

- 将 Section 5.7 从计划性描述改写为基于实际生成结果的结果性描述
- 每行附带定量指标和协议声明

### 3.2 选定的 4 个典型案例

| 行 | 图像 | Failure/Success Mode | 关键指标 |
|---|---|---|---|
| 1 | deepcrack:168 | Text Over-segmentation | Text P=0.266, R=0.971 (高 recall 低 precision) |
| 2 | deepcrack:86 | Point Local Recovery | Point IoU=0.000 vs Text IoU=0.814 (单点仅恢复局部) |
| 3 | deepcrack:17 | AMPF False Negative | Text R=0.813 → AMPF R=0.238, IoU 0.724→0.232 |
| 4 | deepcrack:178 | Mean Fusion Complementary | Text IoU=0.493 → Mean Fusion IoU=0.857 |

### 3.3 图形规格

- 7 列: Original, Ground Truth, Text, Box, Point, AMPF, Mean Fusion
- 4 行（对应 4 个典型样本）
- Mask overlay: 半透明彩色 (alpha=0.45)
- 输出: `experiments/visualizations/fig_qualitative_failure_modes.{png,pdf}`
- 尺寸: 6270×3161px @ 300 DPI

### 3.4 论文中表述

Figure 4 的 caption 和正文明确声明:

- AMPF 和 Mean Fusion 使用 GT-assisted alignment，是 diagnostic protocol
- Text 是唯一 fully automatic 的 protocol
- 4 行分别对应 over-segmentation、local recovery failure、false negatives、complementary recovery

---

## 4. Phase 3: 补充实验 Fig. S1-S3（已完成）

### 4.1 Fig. S1: Confidence Calibration Reliability Diagram

**实验设计**

- 脚本: `experiments/run_candidate_logging.py` + `experiments/plot_calibration.py`
- 方法: 对 DeepCrack test (237 images) 运行 AMPF Stage 1 (independent prompting)，保存每个 candidate 的:
  - `detection_score`: SAM3 原始置信度
  - `stability_score`: 提示扰动一致性
  - `boundary_score`: mask 边界梯度强度
  - `composite_confidence`: α·S_det + β·S_stab + γ·S_bound
  - `candidate_iou`: 与 GT 的 per-candidate IoU
- 输出 CSV: `experiments/results/candidate_confidence.csv` (3449 candidates)
- 耗时: ~64 分钟 (237 images × ~16s/image)

**Reliability Diagram 结构**

- 单个 DeepCrack 子图 (n=237); 当前 candidate log 不含 CrackForest records
- 横轴: mean predicted confidence (10 bins)
- 纵轴: mean candidate IoU
- 对角线: perfect calibration
- 3 条曲线: Text, Box, Point
- 图注标注 ECE

**实验结果**

| Prompt Mode | Candidates | ECE | Corr(conf, IoU) | Mean Conf | Mean IoU |
|---|---|---|---|---|---|
| Text | 2029 | 0.287 | 0.644 | 0.474 | 0.187 |
| Box | 497 | 0.362 | 0.819 | 0.706 | 0.344 |
| Point | 923 | 0.469 | 0.492 | 0.752 | 0.283 |

**关键发现**

1. 所有模式的 ECE 均显著偏离完美校准 (ECE 0.29-0.47)
2. 置信度系统性地高估了实际 IoU: mean confidence 比 mean IoU 高 0.29-0.47
3. Box 的 correlation 最高 (r=0.82) 但 ECE 仍然较高 (0.36)
4. Point 的校准最差 (ECE=0.47)，confidence 最高但 IoU 最低

**论文可写结论**: "On DeepCrack candidate-level diagnostics, composite confidence scores systematically overestimate per-candidate IoU. The Expected Calibration Error ranges from 0.29 (text) to 0.47 (point), indicating that the current confidence-weighting scheme is not reliably calibrated for this visible surface-crack benchmark. This evidence supports the ablation finding that removing components of the confidence-weighted fusion rule does not degrade—and sometimes improves—foreground IoU."

### 4.2 Fig. S2: AMPF Weight Sensitivity Heatmap

**实验设计**

- 脚本: `experiments/run_weight_sensitivity.py` + `experiments/plot_weight_sensitivity.py`
- 设计: 两阶段缓存优化
  - Phase 1: 对 30 张 DeepCrack test 图像运行完整 SAM3 推理（text/box/point + stability/boundary scores），缓存到 pickle 文件
  - Phase 2: 对每个 (α, β) 组合，仅用 NumPy 重算 composite_confidence 和融合（无 SAM3 推理）
- 网格: α ∈ {0.1, 0.2, ..., 0.7}, β ∈ {0.1, 0.2, ..., 0.7}, γ = 1-α-β ≥ 0
- 有效组合: 39 个 (包含 γ=0 boundary combinations)
- 耗时: Phase 1 ~11 分钟 (30 images), Phase 2 <1 秒

**Heatmap 结构**

- triangular heatmap over valid (α, β) region
- colormap: viridis, 颜色表示 DeepCrack mean foreground IoU
- 标注当前 paper 配置 (α=0.4, β=0.35, γ=0.25)
- 标注 best combo
- 图注说明 γ = 1-α-β；γ=0 边界组合用于网格完整性

**实验结果**

| 指标 | 值 |
|---|---|
| IoU 范围 | 0.5303 — 0.5484 |
| IoU 标准差 | 0.0044 |
| 变异系数 (CV) | 0.82% |
| Best combo | α=0.5, β=0.1, γ=0.4 (IoU=0.5484) |
| Corr(IoU, α) | +0.090 (detection weight 影响较弱) |
| Corr(IoU, β) | -0.716 (stability weight 与 IoU 负相关) |
| Corr(IoU, γ) | +0.541 (boundary weight 与 IoU 正相关) |
| 附近网格点 (α=0.4, β=0.3) | IoU=0.5386 |
| 附近网格点 (α=0.4, β=0.4) | IoU=0.5367 |

**关键发现**

1. **Broad plateau**: 在 30-image DeepCrack sensitivity subset 中，IoU 变异系数仅 0.82%，AMPF 对权重配置不敏感
2. **β 的负面影响**: 稳定性权重与 IoU 呈 -0.716 负相关——增加稳定性权重会降低性能。这与 ablation 中 "removing stability improves IoU" 一致
3. **γ 的正面影响**: 边界梯度分数是三项中最有用的置信度分量
4. **当前配置在 plateau 内**: (0.4, 0.35, 0.25) 附近 IoU 约 0.537，与最优值 (0.548) 差距仅 0.011
5. 文本同义词 ("fracture", "fissure", "break") 可能改变了语义而非保持稳定的目标概念，导致 stability score 对于裂缝分割不可靠

**论文可写结论**: "On a 30-image DeepCrack sensitivity subset, weight analysis across 39 (α, β) combinations shows that the foreground IoU varies by less than 1% (CV = 0.82%), forming a broad plateau. The stability weight β is negatively correlated with IoU (r = -0.72), consistent with the ablation result that text-synonym perturbations do not preserve stable crack concepts. The boundary score γ is the most informative confidence component (r = +0.54). These results indicate that, within the tested grid, the AMPF fusion weights themselves are not the primary limitation; the plateau persists even at the best-observed weight configuration."

**Pending (optional)**:
- If needed, can also add CrackForest heatmap, but 24 images likely produce unreliable results
- Can add additional metric heatmaps (Dice, Precision, Recall)

### 4.3 Fig. S3: Inference Time vs Accuracy Profiling

**实验设计**

- 脚本: `experiments/run_profiling.py` + `experiments/plot_runtime_accuracy.py`
- 方法: 对 22 张 DeepCrack test 图像 (2 warmup + 20 timed)，用 `time.perf_counter()` 逐图计时
- 输出 CSV: `experiments/results/runtime_profiling.csv`

**Scatter 结构**

- 横轴: mean ms/image (log scale)
- 纵轴: mean foreground IoU
- 颜色: protocol type (green=deployable, orange=GT-derived diagnostic, red=GT-assisted diagnostic)
- 误差棒: ±1 std ms/image

**实验结果**

| Method | Protocol | Mean ms/img | Std ms/img | Mean IoU |
|---|---|---|---|---|
| Text | automatic text prompt | 84.3 | 7.8 | 0.680 |
| Box | GT-derived prompt | 74.9 | 0.9 | 0.370 |
| Point | GT-derived prompt | 2990.8 | 4570.9 | 0.285 |
| AMPF | GT-assisted fusion | 8866.2 | 9097.8 | 0.530 |
| Mean Fusion | GT-assisted fusion | 3248.0 | 4693.0 | 0.563 |

**关键发现**

1. **Text 是 accuracy-efficiency sweet spot**: 84 ms/img + IoU 0.68，远超其他 SAM3 方法
2. **Box 最快但不准**: 75 ms/img 但 IoU 仅 0.37（全局前景 box 包含大量背景）
3. **Point 效率最差**: ~3s/img 且 IoU 最低 (0.29)，因为每个 GT connected component 需要单独 point-prompt 调用
4. **AMPF 最慢**: ~9s/img，主要开销来自 stability perturbations (每个 candidate 需额外 5-14 次推理)
5. **Mean Fusion 比 AMPF 快 2.7×**: 因为不需要 stability 计算，仅用 alignment + unweighted average fusion
6. 高方差主要来自不同图像 GT 实例数量差异（多实例图像 point prompts 更多）

**论文可写结论**: "On a 20-image DeepCrack profiling subset, text prompting achieves the best accuracy-efficiency trade-off at 84 ms/image with foreground IoU of 0.68. The full AMPF pipeline requires approximately 8.9 s/image due to stability perturbation overhead, while the simpler mean fusion variant reduces this to 3.2 s/image. The large standard deviations reflect variability in the number of ground-truth connected components per image, which determines the number of point prompts; stability re-runs add further overhead for AMPF."

**Pending (optional)**:
- Supervised baselines timing data not collected (training time ≠ inference time)
- Could add U-Net/DeepLabV3+ inference time for completeness

---

## 5. 论文结论更新建议

现在 Fig. S1 和 Fig. S2 已经完成，论文中可以写以下更强的结论（原来不建议写的）：

### 5.1 可以写的强结论（有实验支撑）

1. **"Composite confidence scores are not well calibrated on DeepCrack candidate-level diagnostics."**
   - 支撑: Fig. S1, ECE 0.29-0.47 across modes, confidence systematically overestimates IoU

2. **"The stability perturbation component (β) is negatively correlated with fusion quality in the 30-image DeepCrack sensitivity subset."**
   - 支撑: Fig. S2, r = -0.72; ablation no_stability versus AMPF

3. **"AMPF performance is insensitive to the specific choice of fusion weights within the tested DeepCrack grid."**
   - 支撑: Fig. S2, CV = 0.82% across 39 weight combinations

4. **"Text-prompt inference is the best accuracy-efficiency trade-off among the profiled SAM3 protocols."**
   - 支撑: Fig. S3

### 5.2 仍然不应写的内容

1. "Text prompt is generally deployable for ALL smart infrastructure crack inspection."
   → 限定为 visible surface-crack benchmarks
2. "The method is ready for operational deployment."
   → 仍需 broader validation
3. "Mean Fusion outperforms supervised models."
   → 它是 GT-assisted diagnostic，不可与 deployable 方法直接比较

### 5.3 建议正文调整

- Section 5.4 (Ablation) 可引用 Fig. S2 的权重敏感性结果，加强 "stability 有害" 的论证
- Section 5.5 (Statistical Comparisons) 可补充 ECE 作为 calibration 的定量指标
- Discussion 可引用 Fig. S1 的 ECE 值，支撑 "confidence weighting not calibrated" 的论断

---

## 6. 最终输出文件清单

### 主文 Figures

| 文件 | 大小 |
|---|---|
| `experiments/visualizations/fig_protocol_hierarchy.png` | 301K |
| `experiments/visualizations/fig_protocol_hierarchy.pdf` | 35K |
| `experiments/visualizations/fig_iou_distribution.png` | 705K |
| `experiments/visualizations/fig_iou_distribution.pdf` | 90K |
| `experiments/visualizations/fig_precision_recall_operating_points.png` | 325K |
| `experiments/visualizations/fig_precision_recall_operating_points.pdf` | 33K |
| `experiments/visualizations/fig_qualitative_failure_modes.png` | 3.4M |
| `experiments/visualizations/fig_qualitative_failure_modes.pdf` | 3.4M |

### Supplementary Figures

| 文件 | 大小 |
|---|---|
| `experiments/visualizations/fig_calibration_reliability.png` | 223K |
| `experiments/visualizations/fig_calibration_reliability.pdf` | 27K |
| `experiments/visualizations/fig_ampf_weight_sensitivity_heatmap.png` | 191K |
| `experiments/visualizations/fig_ampf_weight_sensitivity_heatmap.pdf` | 60K |
| `experiments/visualizations/fig_runtime_accuracy_scatter.png` | 164K |
| `experiments/visualizations/fig_runtime_accuracy_scatter.pdf` | 18K |

### 结果数据文件

| 文件 | 内容 |
|---|---|
| `experiments/results/all_results_v2.csv` | 4176 per-image rows, all methods |
| `experiments/results/candidate_confidence.csv` | 3449 DeepCrack candidates, 237 images, per-candidate scores |
| `experiments/results/ampf_weight_sensitivity.csv` | 39 weight combos on 30 DeepCrack images, mean metrics |
| `experiments/results/runtime_profiling.csv` | 5 methods on 20 timed DeepCrack images, ms/image + IoU |

### 新增脚本

| 脚本 | 用途 |
|---|---|
| `experiments/find_representative_cases.py` | 从 CSV 自动筛选 Fig. 4 典型案例 |
| `experiments/generate_qualitative_figure.py` | 生成 Fig. 4 qualitative overlay |
| `experiments/run_candidate_logging.py` | 保存 per-candidate confidence/IoU (Fig. S1) |
| `experiments/plot_calibration.py` | 绘制 reliability diagram + ECE (Fig. S1) |
| `experiments/run_weight_sensitivity.py` | 缓存式权重网格搜索 (Fig. S2) |
| `experiments/plot_weight_sensitivity.py` | 绘制 weight sensitivity heatmap (Fig. S2) |
| `experiments/run_profiling.py` | 逐图推理计时 (Fig. S3) |
| `experiments/plot_runtime_accuracy.py` | 绘制 accuracy-time scatter (Fig. S3) |
| `experiments/plot_protocol_hierarchy.py` | Fig. 1 |
| `experiments/plot_iou_distribution.py` | Fig. 2 |
| `experiments/plot_precision_recall.py` | Fig. 3 |

---

## 7. 制图规范（已实施）

1. ✅ 同时输出 PNG 和 PDF
2. ✅ 使用 colorblind-friendly palette (tab10, viridis)
3. ✅ 所有图中统一方法颜色:
   - Text: blue (#1f77b4)
   - Box: orange (#ff7f0e)
   - Point: purple (#9467bd)
   - AMPF: red (#d62728)
   - Mean Fusion: green/teal
   - supervised baselines: gray (#7f7f7f)
4. ✅ 所有图标注 dataset
5. ✅ GT-assisted 方法在图例中标注 `diagnostic` 或 `GT-assisted`
6. ✅ Qualitative mask overlay alpha=0.45
7. ✅ 图注中声明 Mean Fusion / AMPF 的 protocol caveat
8. ✅ 所有图 300 DPI

---

## 8. 完整任务清单

### Phase 1: 主文 Fig. 1-3 ✅ (2026-05-27)

- [x] 画 Fig. 1 protocol hierarchy
- [x] 画 Fig. 2 per-image IoU distribution
- [x] 画 Fig. 3 precision-recall scatter
- [x] 正文引用 Fig. 1-3
- [x] 所有涉及 AMPF / Mean Fusion 的图注显式标注 `GT-assisted diagnostic`
- [x] 保留并完善 visible surface-crack benchmark 限定

### Phase 2: Fig. 4 Qualitative Analysis ✅ (2026-05-27)

- [x] 写 `find_representative_cases.py` 自动筛选典型案例
- [x] 运行 `generate_qualitative_figure.py` 生成 4 行 × 7 列 overlay 面板
- [x] 修改 `manuscript_revised_protocol_aware.md` Section 5.7 为结果性描述
- [x] 更新 Data and Code Availability 章节

### Phase 3: Supplementary Figs S1-S3 ✅ (2026-05-27/28)

- [x] `run_candidate_logging.py` — 3449 candidates, 237 images
- [x] `plot_calibration.py` — reliability diagram + ECE
- [x] `run_weight_sensitivity.py` — 39 weight combos, 30 images, caching optimization
- [x] `plot_weight_sensitivity.py` — triangular heatmap
- [x] `run_profiling.py` — 5 methods, per-image timing
- [x] `plot_runtime_accuracy.py` — accuracy-efficiency scatter

### 论文最终整合 (建议下一步)

- [ ] 将 Fig. S1-S3 的发现整合进 Discussion/Limitations
- [ ] 可选: 将 S1 的强 evidence 升级到主文
- [ ] 可选: 补充 supervised baseline inference time
- [ ] 终稿 proofreading
