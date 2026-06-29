# FGBind-Net

FGBind-Net is a foreground-background guided sequence-level model for
protein-ligand binding affinity prediction.

This repository contains the model checkpoint, processed benchmark feature
tables, and evaluation script used for FGBind-Net prediction on CASF-2013 and
CASF-2016.

## Repository Contents

```text
fgbind_net_inference.pt     Model checkpoint
evaluate_fgbind.py          Evaluation script
requirements.txt            Python environment

data/train.csv              Training set
data/validation.csv         Validation set
data/casf2013.csv           CASF-2013 test set
data/casf2016.csv           CASF-2016 test set
```

## Requirements

The software environment is listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Data format

Each file in `data/` is provided in CSV format with the following columns:

```text
pdb_id,affinity,full_protein_seq,fg_pocket_seq,bg_protein_seq,ligand_smiles
```

Column descriptions:

- `pdb_id`: complex identifier.
- `affinity`: experimental binding affinity value.
- `full_protein_seq`: full protein sequence.
- `fg_pocket_seq`: binding-pocket sequence used as foreground input.
- `bg_protein_seq`: non-pocket protein sequence used as background input.
- `ligand_smiles`: ligand SMILES string.

The processed data were generated from PDBbind v2020 and the CASF benchmark
sets. Original PDBbind resources should be cited and used according to their
terms of use.

## Evaluation

Run evaluation on CASF-2013 and CASF-2016:

```bash
python evaluate_fgbind.py \
  --model fgbind_net_inference.pt \
  --input data/casf2013.csv data/casf2016.csv \
  --output-dir results
```

The script writes prediction files and a metric summary:

```text
results/casf2013_predictions.csv
results/casf2016_predictions.csv
results/metrics_summary.csv
```

The summary contains five metrics: RMSE, PCC, MAE, SD and CI.

## Model input lengths

The released inference model uses the following maximum lengths:

- background protein sequence: 819
- foreground pocket sequence: 66
- ligand SMILES: 153

Longer inputs are truncated to the maximum length. Shorter inputs are padded
with the mask token. Characters outside the predefined vocabulary are mapped to
the mask token.

## Citation

If you use FGBind-Net, please cite:

FGBind-Net: foreground-background guided local-global modeling for
protein-ligand binding affinity prediction.
