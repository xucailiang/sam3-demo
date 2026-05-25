# 实验完整记录 — SAM3 Multi-Prompt Fusion for Crack Segmentation

> 实验日期：2026-05-21
> 数据来源：`experiments/results/all_results.csv`（4176 行）
> 代码版本：`sam3-demo` @ commit `a035474`

---

## 1. 硬件与环境

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 3090 (24GB VRAM) |
| Driver | 580.142, CUDA 13.0 |
| CPU | AMD Ryzen (Fedora 42) |
| RAM | 32GB |
| Python | 3.13.13 |
| PyTorch | 2.11.0+cu130 |
| Ultralytics | 8.4.33 |
| SAM3 模型 | `sam3.pt`，half-precision (fp16) |
| 虚拟环境 | `sam3-demo/.venv` (uv, Python 3.11) |

---

## 2. 数据集

| 数据集 | 来源 | 图像数 | 分辨率 | 训练/验证/测试 |
|--------|------|--------|--------|----------------|
| CrackForest | [cuilimeng/CrackForest-dataset](https://github.com/cuilimeng/CrackForest-dataset) | 118 | 480×320 | 60/20/20 (seed=42) |
| DeepCrack | [yhlleo/DeepCrack](https://github.com/yhlleo/DeepCrack) | 537 | 多尺度 | 240/60/237 (官方划分) |

- CrackForest: 路面裂缝（单一场景，类间变异性低）
- DeepCrack: 路面+桥梁+混凝土裂缝（多场景，类间变异性高）
- 存储路径：`sam3-demo/experiments/datasets/`

---

## 3. 实验设计

### 3.1 实验协议 (Protocol)

| 协议 | 含义 | 使用场景 |
|------|------|---------|
| `automatic_text_prompt` | 使用固定文本 "crack, fracture, fissure, break" | Text 单模式 |
| `oracle_gt_prompt` | 从 GT mask 生成 box/point prompt | Box/Point 单模式 |
| `oracle_gt_prompt_gt_alignment` | Oracle prompt + 实例级 GT 对齐 | AMPF, Ablation, Mode Combo |
| `supervised_train_val_test` | 全监督训练 (train/val/test 三集) | U-Net, DeepLabV3+, YOLOv8-seg |

### 3.2 13 组实验（每数据集）

| # | 组名 | 协议 | 说明 |
|---|------|------|------|
| 1 | `text` | automatic_text_prompt | Text 单模式基线 |
| 2 | `box` | oracle_gt_prompt | Box 单模式基线 |
| 3 | `point` | oracle_gt_prompt | Point 单模式基线 |
| 4 | `ampf` | oracle_gt_prompt_gt_alignment | AMPF 全管线（text+box+point 融合） |
| 5 | `ablation_no_detection` | oracle_gt_prompt_gt_alignment | 去掉检测置信度 S_det |
| 6 | `ablation_no_stability` | oracle_gt_prompt_gt_alignment | 去掉稳定性评分 S_stab |
| 7 | `ablation_no_boundary` | oracle_gt_prompt_gt_alignment | 去掉边界评分 S_bnd |
| 8 | `ablation_no_alignment` | oracle_gt_prompt_gt_alignment | 去掉实例级对齐（关键消融） |
| 9 | `ablation_fusion_max` | oracle_gt_prompt_gt_alignment | 融合策略 = max |
| 10 | `ablation_fusion_mean` | oracle_gt_prompt_gt_alignment | 融合策略 = 等权平均 |
| 11 | `combo_text+box` | oracle_gt_prompt_gt_alignment | 双模态：Text+Box |
| 12 | `combo_text+point` | oracle_gt_prompt_gt_alignment | 双模态：Text+Point |
| 13 | `combo_box+point` | oracle_gt_prompt_gt_alignment | 双模态：Box+Point |

### 3.3 监督基线（100 epochs, AdamW, lr=1e-4）

| 方法 | 架构 | 参数量 |
|------|------|--------|
| U-Net | SMP (EfficientNet-b0 backbone) | ~5.7M |
| DeepLabV3+ | SMP (ResNet-50 backbone) | ~26.7M |
| YOLOv8-seg | Ultralytics YOLOv8n-seg | ~3.3M |

---

## 4. 实验结果

### 4.1 主结果 — 单模式 vs AMPF

| 方法 | CrackForest mIoU | CrackForest Dice | DeepCrack mIoU | DeepCrack Dice |
|------|-----------------|-----------------|----------------|----------------|
| **Text** | **0.4355** | 0.6000 | **0.6658** | 0.7866 |
| Box | 0.3439 | 0.4786 | 0.5408 | 0.6441 |
| Point | 0.1166 | 0.1768 | 0.3129 | 0.3860 |
| AMPF (Ours) | 0.4249 | 0.5843 | 0.6645 | 0.7854 |

**发现：** Text 单模态在两个数据集上均优于或持平 AMPF。Box 和 Point 融合非但没有提升，反而拉低了 Text 的质量。

### 4.2 消融实验 — AMPF 组件贡献

| 消融 | CrackForest mIoU | DeepCrack mIoU | vs AMPF (DC) |
|------|-----------------|----------------|-------------|
| AMPF (full) | 0.4249 | 0.6645 | — |
| w/o S_det | 0.4352 | 0.6702 | +0.0057 |
| w/o S_stab | 0.4382 | 0.6694 | +0.0049 |
| w/o S_bnd | 0.4215 | 0.6589 | -0.0056 |
| **w/o alignment** | **0.3285** | **0.4418** | **-0.2227** |
| fusion=max | 0.4276 | 0.6664 | +0.0019 |
| **fusion=mean** | **0.4535** | **0.6791** | **+0.0146** |

**发现：**
- **Alignment 是关键**：去掉后 DeepCrack 下降 33.5%（0.6645 → 0.4418）
- **等权平均 > 置信度加权**：fusion=mean 在两个数据集均优于 AMPF
- 三个评分组件贡献微弱且方向不一致

### 4.3 双模态组合 vs 三模态 AMPF

| 方法 | CrackForest mIoU | DeepCrack mIoU |
|------|-----------------|----------------|
| Text | 0.4355 | 0.6658 |
| Box+Text | 0.4281 | 0.6658 |
| Point+Text | 0.4257 | 0.6523 |
| Box+Point | 0.4075 | 0.6202 |
| AMPF (all three) | 0.4249 | 0.6645 |

**发现：** 任何包含 Point 或 Box 的组合都 ≤ Text alone。双模态最佳 = Text alone。

### 4.4 零训练 vs 全监督

| 方法 | CrackForest mIoU | DeepCrack mIoU | 训练需求 |
|------|-----------------|----------------|---------|
| Text (zero-training) | 0.4355 | 0.6658 | 无 |
| AMPF (zero-training) | 0.4249 | 0.6645 | 无 |
| fusion=mean (zero) | 0.4535 | 0.6791 | 无 |
| **U-Net** (100 epochs) | **0.4576** | 0.6427 | 标注数据 |
| DeepLabV3+ (100 epochs) | 0.4384 | 0.6298 | 标注数据 |
| YOLOv8-seg (100 epochs) | 0.3205 | 0.4155 | 标注数据 |

**发现：** DeepCrack 上 Text (0.666) > U-Net (0.643)；CrackForest 上 U-Net (0.458) 略胜 Text (0.436)。

### 4.5 DeepCrack 完整结果总表（237 张，各组均含 mIoU/Dice/Precision/Recall/F1）

| 方法 | mIoU | Dice | Precision | Recall | F1 |
|------|------|------|-----------|--------|-----|
| text | 0.6658 | 0.7866 | 0.7441 | 0.8854 | 0.7866 |
| box | 0.5408 | 0.6441 | 0.6603 | 0.7516 | 0.6441 |
| point | 0.3129 | 0.3860 | 0.3584 | 0.6363 | 0.3860 |
| ampf | 0.6645 | 0.7854 | 0.8118 | 0.8106 | 0.7854 |
| no_detection | 0.6702 | 0.7907 | 0.8176 | 0.8133 | 0.7907 |
| no_stability | 0.6694 | 0.7897 | 0.8147 | 0.8149 | 0.7897 |
| no_boundary | 0.6589 | 0.7805 | 0.8017 | 0.8134 | 0.7805 |
| no_alignment | 0.4418 | 0.5205 | 0.6487 | 0.5150 | 0.5205 |
| fusion_max | 0.6664 | 0.7866 | 0.7660 | 0.8693 | 0.7866 |
| **fusion_mean** | **0.6791** | **0.7985** | 0.8073 | 0.8365 | 0.7985 |
| text+box | 0.6658 | 0.7864 | 0.8180 | 0.8033 | 0.7864 |
| text+point | 0.6523 | 0.7755 | 0.7901 | 0.8189 | 0.7755 |
| box+point | 0.6202 | 0.7360 | 0.7920 | 0.7535 | 0.7360 |
| U-Net (sup.) | 0.6427 | 0.7637 | 0.8521 | 0.7359 | 0.7637 |
| DeepLabV3+ | 0.6298 | 0.7555 | 0.8463 | 0.7242 | 0.7555 |
| YOLOv8-seg | 0.4155 | 0.5616 | 0.6102 | 0.5436 | 0.5616 |

> 完整 per-image 数据：`experiments/results/all_results.csv`（4176 行）

---

## 5. 运行耗时

| 阶段 | 进程数 | 耗时 | 备注 |
|------|--------|------|------|
| SAM3/AMPF CrackForest (24张, 13组) | 1 | ~12 min | — |
| SAM3/AMPF DeepCrack (50张, 13组) | 1 | ~45 min | — |
| 监督基线 CrackForest (3 models) | 1 | ~17 min | 100 epochs each |
| 监督基线 DeepCrack (3 models) | 1 | ~9 min | 100 epochs each |
| **CrackForest + DeepCrack (50张) 全量** | 1 | **73.8 min** | `run_all.py` |
| DeepCrack full (237张) groups 1-3 | 1 | ~89 min | text/box/point |
| DeepCrack full (237张) groups 4-13 | 1 | **264.9 min** | ampf + 6 ablation + 3 combo |
| **总实验时间** | — | **~7 小时** | 单进程安全模式 |

> ⚠️ 多进程并行遇到 CUDA 死锁（YOLO multiprocessing），最终改用单进程顺序执行。

---

## 6. 核心发现

1. **Text alone 就够了**：DeepCrack 上 Text mIoU=0.666，超越全监督 U-Net (0.643)；AMPF 相对 Text 的边际改善为负或零
2. **Alignment 是瓶颈**：去掉实例级对齐后，mIoU 下降 25-33%（CrackForest: 0.425→0.329, DeepCrack: 0.665→0.442）
3. **简单融合 > 复杂加权**：等权平均 (`fusion=mean`) 在两个数据集均优于置信度加权 AMPF
4. **Point 不适合裂缝**：Point 单模态 mIoU 仅 0.117/0.313，且加入任何融合组合都会降低性能
5. **Precision-Recall 权衡**：AMPF 提升 Precision（0.744→0.812 on DC），但降低 Recall（0.885→0.811）——融合引入假阴性
6. **零训练潜力**：无需任何裂缝标注数据即可获得与 100-epoch 监督训练相当的 mIoU

---

## 7. 可选的论文题目

### 推荐（Systematic Evaluation 型）

> **How Much Does Multi-Prompt Fusion Help? A Systematic Evaluation of SAM3-Based Zero-Training Crack Segmentation**

适合 MDPI Applied Sciences special issue "AI-Driven Urban Development and Smart Infrastructure"。
叙事：诚实的负结果论文——系统评估多 prompt 融合，发现 text alone 已足够好。

### 备选

| # | 题目 | 定位 |
|---|------|------|
| 1 | *Evaluating Multi-Prompt Fusion Strategies for SAM3-Based Zero-Training Crack Segmentation in Smart Infrastructure Inspection* | 应用方法型 |
| 2 | *When Less Is More: Text Prompting Alone Outperforms Multi-Prompt Fusion for SAM3-Based Crack Detection* | 发现导向型 |
| 3 | *Instance-Level Alignment as a Prerequisite for Multi-Prompt Fusion in Vision Foundation Models* | 方法论型 |
| 4 | *Zero-Training Crack Segmentation with SAM3: A Benchmark of Prompt Modes, Fusion Strategies, and Supervised Baselines* | 基准测试型 |

### 中文题目

> 面向智慧基础设施巡检的 SAM3 多提示融合策略评估与裂缝分割方法

---

## 8. 复现步骤

```bash
cd sam3-demo
uv venv .venv --python 3.11
uv pip install -r backend/requirements.txt scipy pandas matplotlib \
    segmentation-models-pytorch --python .venv/bin/python

# 全量实验
.venv/bin/python experiments/run_all.py --no-baselines
.venv/bin/python experiments/run_baselines.py --dataset crackforest
.venv/bin/python experiments/run_baselines.py --dataset deepcrack
.venv/bin/python experiments/run_deepcrack_full.py --start 1 --end 13

# 生成论文材料
.venv/bin/python experiments/generate_tables.py
.venv/bin/python experiments/generate_plots.py
```

---

## 9. 项目文件结构

```
/home/justin/workspace/cv_paper/
├── README.md                            # 项目概览
├── EXPERIMENTS.md                       # ← 本文件（实验完整记录）
├── paper.md                             # 研究计划书
├── paper_results/                       # 论文输出材料
│   ├── all_results.csv                  # 4176 行全量结果
│   ├── summary_stats.csv                # 汇总统计
│   ├── paper_tables.tex                 # 4 个 LaTeX 表格
│   ├── fig_main_comparison.png          # 主对比图
│   ├── fig_ablation.png                 # 消融实验图
│   ├── fig_supervised_comparison.png    # 监督对比图
│   └── fig_mode_combinations.png        # 模态组合图
├── experiments/                         # 实验代码与结果副本
│   ├── results/                         # 所有独立 CSV 文件
│   ├── visualizations/                  # 对比图和指标柱状图
│   ├── datasets/                        # CrackForest + DeepCrack
│   ├── run_all.py                       # 全量实验入口
│   ├── run_deepcrack_full.py            # DeepCrack 237 张实验
│   ├── run_baselines.py                 # 监督基线训练+评估
│   ├── generate_tables.py               # LaTeX 表格生成
│   ├── generate_plots.py                # matplotlib 图表生成
│   └── *.log                            # 完整运行日志
└── sam3-demo/                           # 上游代码仓库
    ├── backend/app/services/
    │   ├── ampf_engine.py               # AMPF 三阶段融合引擎
    │   ├── segmentation_service.py      # SAM3 推理服务
    │   └── model_manager.py             # SAM3 + YOLO 模型管理
    └── experiments/                     # 原始实验目录
```
