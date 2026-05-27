# 实验完整记录 — SAM3 Multi-Prompt Fusion for Crack Segmentation

> 实验日期：2026-05-21；point protocol 修正与 DeepCrack 重跑：2026-05-26
> 数据来源：`experiments/results/all_results_v2.csv`（4176 条 per-image 结果；CSV 含表头共 4177 行）
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
| `automatic_text_prompt` | 使用固定文本 "crack" | Text 单模式 |
| `oracle_gt_prompt` | 从 GT mask 生成 box/point prompt；point 使用 nearest-foreground protocol | Box/Point 单模式 |
| `oracle_gt_prompt_gt_alignment` | Oracle prompt + 实例级 GT 对齐 | AMPF, Ablation, Mode Combo |
| `supervised_train_val_test` | 全监督训练 (train/val/test 三集) | U-Net, DeepLabV3+, YOLOv8-seg |

> **注**: Text 单模式 baseline 使用固定 prompt `"crack"`。同义词 `"fracture"`, `"fissure"`, `"break"` 仅在 AMPF 的 text-prompt stability scoring 中使用，不用于 text single-mode baseline。

### 3.2 13 组实验（每数据集）

| # | 组名 | 协议 | 说明 |
|---|------|------|------|
| 1 | `text` | automatic_text_prompt | Text 单模式基线 |
| 2 | `box` | oracle_gt_prompt | Box 单模式基线 |
| 3 | `point` | oracle_gt_prompt | Point 单模式基线（每个 GT 连通域一个 nearest-foreground positive point） |
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

### 3.3 监督基线（U-Net/DeepLabV3+: 50 epochs, Adam, lr=1e-4; YOLOv8-seg: 100 epochs）

| 方法 | 架构 | 参数量 |
|------|------|--------|
| U-Net | SMP (ResNet-34 backbone, ImageNet pretrained) | ~24M |
| DeepLabV3+ | SMP (ResNet-34 backbone, ImageNet pretrained) | ~27M |
| YOLOv8-seg | Ultralytics YOLOv8n-seg | ~3.3M |

---

## 4. 实验结果

### 4.1 主结果 — 单模式 vs AMPF

| 方法 | CrackForest IoU | CrackForest Dice | DeepCrack IoU | DeepCrack Dice |
|------|-----------------|-----------------|----------------|----------------|
| **Text** | **0.4355** | 0.6000 | **0.6658** | 0.7866 |
| Box | 0.3439 | 0.4786 | 0.5408 | 0.6441 |
| Point (nearest-fg) | 0.3014 | 0.4332 | 0.5199 | 0.6278 |
| AMPF (Ours) | 0.4145 | 0.5737 | 0.6671 | 0.7880 |

**发现：** Text 单模态是最强的可部署单提示基线。nearest-foreground point 相比旧 centroid protocol 大幅提升，但仍弱于 text 和 box；AMPF 在 DeepCrack 上略高于 Text，在 CrackForest 上低于 Text，整体收益不稳定。

### 4.2 消融实验 — AMPF 组件贡献

| 消融 | CrackForest IoU | DeepCrack IoU | vs AMPF (DC) |
|------|-----------------|----------------|-------------|
| AMPF (full) | 0.4145 | 0.6671 | — |
| w/o S_det | 0.4087 | 0.6639 | -0.0032 |
| w/o S_stab | 0.4279 | 0.6691 | +0.0020 |
| w/o S_bnd | 0.4122 | 0.6605 | -0.0066 |
| **w/o alignment** | **0.3506** | **0.4676** | **-0.1995** |
| fusion=max | 0.4084 | 0.6746 | +0.0076 |
| **fusion=mean** | **0.4587** | **0.6815** | **+0.0144** |

**发现：**
- **Alignment 是关键**：去掉后 DeepCrack IoU 下降 0.1995（0.6671 → 0.4676），CrackForest 下降 0.0639（0.4145 → 0.3506）
- **等权平均 > 置信度加权**：fusion=mean 在两个数据集均优于 AMPF
- 三个评分组件贡献微弱且方向不一致

### 4.3 双模态组合 vs 三模态 AMPF

| 方法 | CrackForest IoU | DeepCrack IoU |
|------|-----------------|----------------|
| Text | 0.4355 | 0.6658 |
| Box+Text | 0.4282 | 0.6654 |
| Point+Text | 0.3972 | 0.6435 |
| Box+Point | 0.4012 | 0.6466 |
| AMPF (all three) | 0.4145 | 0.6671 |

**发现：** nearest-foreground point 改善了 point 单模式质量，但含 point 的融合组合仍未稳定超过 Text。DeepCrack 上 AMPF 只比 Text 高 0.0013，CrackForest 上则低于 Text，说明多提示融合的收益依赖对齐与置信度校准。

### 4.4 零训练 vs 全监督

| 方法 | CrackForest IoU | DeepCrack IoU | 训练需求 |
|------|-----------------|----------------|---------|
| Text (zero-training) | 0.4355 | 0.6658 | 无 |
| AMPF (GT-assisted diagnostic) | 0.4145 | 0.6671 | 无训练；使用 GT-assisted prompt/alignment |
| fusion=mean (GT-assisted diagnostic) | 0.4587 | 0.6815 | 无训练；使用 GT-assisted prompt/alignment |
| **U-Net** (50 epochs) | **0.4576** | 0.6661 | 标注数据 |
| DeepLabV3+ (50 epochs) | 0.4384 | 0.6698 | 标注数据 |
| YOLOv8-seg (100 epochs) | 0.3069 | 0.4198 | 标注数据 |

**发现：** DeepCrack 上 Text (0.6658) 与 U-Net (0.6661) 基本持平，略低于 DeepLabV3+ (0.6698)。GT-assisted mean fusion 达到 0.6815，但应作为诊断性结果解读，不等同于可部署零训练方法。

### 4.5 DeepCrack 完整结果总表（237 张，各组均含 IoU/Dice/Precision/Recall/F1）

| 方法 | IoU | Dice | Precision | Recall | F1 |
|------|------|------|-----------|--------|-----|
| text | 0.6658 | 0.7866 | 0.7441 | 0.8854 | 0.7866 |
| box | 0.5408 | 0.6441 | 0.6603 | 0.7516 | 0.6441 |
| point (nearest-fg) | 0.5199 | 0.6278 | 0.6229 | 0.7754 | 0.6278 |
| ampf | 0.6671 | 0.7880 | 0.8126 | 0.8105 | 0.7880 |
| no_detection | 0.6639 | 0.7844 | 0.8126 | 0.8076 | 0.7844 |
| no_stability | 0.6691 | 0.7894 | 0.8149 | 0.8122 | 0.7894 |
| no_boundary | 0.6605 | 0.7825 | 0.8055 | 0.8097 | 0.7825 |
| no_alignment | 0.4676 | 0.5489 | 0.6690 | 0.5436 | 0.5489 |
| fusion_max | 0.6746 | 0.7936 | 0.7640 | 0.8799 | 0.7936 |
| **fusion_mean** | **0.6815** | **0.8012** | 0.8126 | 0.8332 | 0.8012 |
| text+box | 0.6654 | 0.7860 | 0.8178 | 0.8027 | 0.7860 |
| text+point | 0.6435 | 0.7688 | 0.7864 | 0.8134 | 0.7688 |
| box+point | 0.6466 | 0.7645 | 0.8079 | 0.7812 | 0.7645 |
| U-Net (sup.) | 0.6661 | 0.7855 | 0.8122 | 0.8042 | 0.7855 |
| DeepLabV3+ | 0.6698 | 0.7906 | 0.8351 | 0.7815 | 0.7906 |
| YOLOv8-seg | 0.4198 | 0.5652 | 0.6124 | 0.5457 | 0.5652 |

> 完整 per-image 数据：`experiments/results/all_results_v2.csv`（4176 条结果；CSV 含表头共 4177 行）

---

## 5. 运行耗时

| 阶段 | 进程数 | 耗时 | 备注 |
|------|--------|------|------|
| SAM3/AMPF CrackForest (24张, 13组) | 1 | ~12 min | — |
| SAM3/AMPF DeepCrack (50张, 13组) | 1 | ~45 min | — |
| 监督基线 CrackForest (3 models) | 1 | ~17 min | U-Net/DeepLabV3+: 50 epochs; YOLOv8-seg: 100 epochs |
| 监督基线 DeepCrack (3 models) | 1 | ~9 min | U-Net/DeepLabV3+: 50 epochs; YOLOv8-seg: 100 epochs |
| **CrackForest + DeepCrack pilot/full-run entry** | 1 | **73.8 min** | `run_all.py` |
| DeepCrack full (237张) groups 1-3 | 1 | ~89 min | text/box/point |
| DeepCrack full (237张) groups 4-13 | 1 | **264.9 min** | ampf + 6 ablation + 3 combo |
| **总实验时间** | — | **~7 小时** | 单进程安全模式 |

> ⚠️ 多进程并行遇到 CUDA 死锁（YOLO multiprocessing），最终改用单进程顺序执行。

---

## 6. 核心发现

1. **Text alone 是最强可部署基线**：DeepCrack 上 Text IoU=0.6658，与 U-Net (0.6661) 基本持平；AMPF 的可见收益很小且依赖 GT-assisted prompt/alignment
2. **Alignment 是瓶颈**：去掉实例级对齐后，IoU 明显下降（CrackForest: 0.4145→0.3506, DeepCrack: 0.6671→0.4676）
3. **简单融合 > 复杂加权**：等权平均 (`fusion=mean`) 在两个数据集均优于置信度加权 AMPF
4. **Point protocol 修正后仍有限**：nearest-foreground point 将 Point IoU 提升到 0.301/0.520，但单点提示仍弱于 Text/Box，说明问题不只是 centroid 落背景，也包括裂缝细长、分叉和不连续几何
5. **Precision-Recall 权衡**：AMPF 提升 Precision（0.744→0.812 on DC），但降低 Recall（0.885→0.811）——融合引入假阴性
6. **零训练潜力**：无需任何裂缝标注数据即可获得与监督训练相当的 IoU

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
# from the submission project root: sam3-demo/
uv venv .venv --python 3.11
uv pip install -r backend/requirements.txt scipy pandas matplotlib \
    segmentation-models-pytorch --python .venv/bin/python

# 全量实验
.venv/bin/python experiments/run_all.py --no-baselines
.venv/bin/python experiments/run_baselines.py --dataset crackforest
.venv/bin/python experiments/run_baselines.py --dataset deepcrack
.venv/bin/python experiments/run_deepcrack_full.py --start 1 --end 13

# 生成 v2 论文材料
.venv/bin/python experiments/rebuild_paper_materials.py
```

---

## 9. 项目文件结构

```
sam3-demo/
├── README.md                            # 项目概览
├── manuscript.md                        # 论文正文
├── EXPERIMENTS.md                       # 本文件：实验完整记录
├── POINT_PROTOCOL_FIX.md                # nearest-foreground point protocol 修正记录
├── REVISION_NOTES.md                    # 第一轮导师意见修订记录
├── manuscript_review_issues.md          # 导师式预审问题清单
├── experiments/
│   ├── results/
│   │   ├── all_results_v2.csv           # 4176 条 per-image 结果
│   │   ├── summary_stats_v2.csv         # 32 条 per-method 汇总
│   │   ├── paper_tables_v2.tex          # 论文 LaTeX 表格
│   │   └── point_experiments_v2.csv     # point 相关重跑汇总
│   ├── visualizations/                  # v2 对比图和指标柱状图
│   ├── datasets/                        # CrackForest + DeepCrack
│   ├── tests/                           # prompt generation 等测试
│   ├── prompt_generator.py              # box/point/text prompt 生成
│   ├── run_all.py                       # 全量实验入口
│   ├── run_deepcrack_full.py            # DeepCrack 237 张实验
│   ├── run_baselines.py                 # 监督基线训练+评估
│   └── rebuild_paper_materials.py       # v2 结果合并、LaTeX 表格与图表重建
└── backend/app/services/
    ├── ampf_engine.py                   # AMPF 三阶段融合引擎
    ├── segmentation_service.py          # SAM3 推理服务
    └── model_manager.py                 # SAM3 + YOLO 模型管理
```
