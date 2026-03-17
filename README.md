# TREX

TREX is a Python-based machine learning pipeline for T cell receptor (TCR) classification using V/J gene usage and CDR3 sequence features. The repository includes scripts for data cleaning, similarity-based sample filtering, feature encoding, and comparative model training.

---

## Overview

This project is designed to:

- preprocess raw TCR trait data from Excel files
- remove highly similar or potentially confounding samples
- encode categorical and sequence-derived features
- train and compare multiple machine learning models
- export evaluation results and preprocessing artifacts

The current workflow uses the following feature groups:

- **V/J gene features**: `TRAV`, `TRAJ`, `TRBV`, `TRBJ`
- **CDR3 sequence features**: `CDR3a`, `CDR3b`

The primary label column is:

- `Label`

---

## Repository Structure

```text
TREX/
├── clean_data.py
├── compared.py
├── deldignosed.py
├── environment.yml
├── health.xlsx
├── last_comparedcode.py
├── test.py
├── train_onlyCDR.py
├── train_VJcompare.py
├── train_XGboost.py
├── trait_train.xlsx
├── trait_train_cleaned.xlsx
├── model_comparison_20260206_091003/
└── README.md
```

------

## Usage

## Environment Setup

Create the Conda environment from the provided file:

```bash
conda env create -f environment.yml -n xgboost
conda activate xgboost
```



### Step 1. Clean the dataset

```bash
python deldignosed.py
```

### Step 2. Train and compare models

```bash
python train_XGboost.py
```

------

## Output

The training script creates a timestamped result directory, for example:

```text
model_comparison_YYYYMMDD_HHMMSS/
```

Outputs may include:

- trained preprocessing objects
- model evaluation summaries
- ROC / PR related figures
- fold-wise and aggregated metrics

------

## Methods Summary

The current implementation combines:

- categorical gene usage features
- position-wise encoding of CDR3 amino acid sequences
- one-hot feature transformation
- classical machine learning baselines
- cross-validation and bootstrap-based uncertainty estimation

This design allows fast comparison of multiple classifiers on TCR-derived features.

------

## Notes

- The scripts currently use Excel input files directly.
- Some paths and parameters may be hard-coded and need adjustment before reuse on another machine.
- `deldignosed.py` may require a local ESM2 model path to be updated before running.
- The repository is still being refined, so script names and workflow details may continue to evolve.

------

## Future Improvements

Potential future improvements include:

- adding command-line arguments instead of hard-coded file paths
- standardizing script naming
- adding a reproducible `results/` layout
- documenting each experiment setting separately
- adding example input data and expected output files
- providing a figure of the full workflow

------

