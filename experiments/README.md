# Experiments

## Dataset Setup

Datasets are intentionally ignored by git. Prepare them on a new machine with:

```bash
cd sam3-demo
bash experiments/prepare_datasets.sh
```

Expected layout:

```text
experiments/datasets/
  CrackForest-dataset/
    image/
    groundTruth/
  DeepCrack/
    dataset/
      DeepCrack/
        train_img/
        train_lab/
        test_img/
        test_lab/
```

Validate an existing local dataset copy with:

```bash
.venv/bin/python experiments/validate_datasets.py
```

Expected sample counts:

- CrackForest: 118 labeled image-mask pairs
- DeepCrack: 537 labeled image-mask pairs

## Protocol Reminder

Experiment CSVs include a `protocol` column. Keep this column in paper tables:

- `automatic_text_prompt`: deployable text-only prompt baseline
- `oracle_gt_prompt`: GT-generated box/point prompt upper bound
- `oracle_gt_prompt_gt_alignment`: GT-generated prompts plus GT-assisted AMPF alignment
- `supervised_train_val_test`: supervised train/validation/test baseline
