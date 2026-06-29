import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


SMI_CHAR_SET = {
    "<MASK>": 0, "C": 1, ")": 2, "(": 3, "c": 4, "O": 5, "]": 6, "[": 7,
    "@": 8, "1": 9, "=": 10, "H": 11, "N": 12, "2": 13, "n": 14,
    "3": 15, "o": 16, "+": 17, "-": 18, "S": 19, "F": 20, "p": 21,
    "l": 22, "/": 23, "4": 24, "#": 25, "B": 26, "\\": 27, "5": 28,
    "r": 29, "s": 30, "6": 31, "I": 32, "7": 33, "%": 34, "8": 35,
    "e": 36, "P": 37, "9": 38, "R": 39, "u": 40, "0": 41, "i": 42,
    ".": 43, "A": 44, "t": 45, "h": 46, "V": 47, "g": 48, "b": 49,
    "Z": 50, "T": 51, "M": 52,
}

SEQ_CHAR_SET = {
    "<MASK>": 0, "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
    "H": 7, "K": 8, "I": 9, "L": 10, "M": 11, "N": 12, "P": 13,
    "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "Y": 19, "W": 20,
}

PROTEIN_SEQ_LEN = 819
POCKET_SEQ_LEN = 66
SMI_LEN = 153


def encode_text(value, vocab, max_len):
    encoded = np.zeros(max_len, dtype=np.int64)
    if not isinstance(value, str):
        return encoded
    for i, token in enumerate(value[:max_len]):
        encoded[i] = vocab.get(token, 0)
    return encoded


class FGBindDataset(Dataset):
    required_columns = [
        "pdb_id",
        "ligand_smiles",
        "full_protein_seq",
        "fg_pocket_seq",
        "bg_protein_seq",
    ]

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        missing = [col for col in self.required_columns if col not in self.data.columns]
        if missing:
            raise ValueError(
                f"{self.csv_path} is missing required columns: " + ", ".join(missing)
            )
        self.has_affinity = "affinity" in self.data.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        pdb_id = str(row["pdb_id"])
        smi = torch.tensor(encode_text(row["ligand_smiles"], SMI_CHAR_SET, SMI_LEN)).long()
        bg_seq = torch.tensor(
            encode_text(row["bg_protein_seq"], SEQ_CHAR_SET, PROTEIN_SEQ_LEN)
        ).long()
        pocket_seq = torch.tensor(
            encode_text(row["fg_pocket_seq"], SEQ_CHAR_SET, POCKET_SEQ_LEN)
        ).long()
        mask_pocket = torch.zeros(PROTEIN_SEQ_LEN, dtype=torch.long)
        if self.has_affinity:
            affinity = float(row["affinity"])
        else:
            affinity = np.nan
        return pdb_id, pocket_seq, smi, bg_seq, mask_pocket, affinity


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def concordance_index(y_true, y_pred):
    concordant = 0.0
    comparable = 0
    for i in range(1, len(y_true)):
        for j in range(i):
            if y_true[i] == y_true[j]:
                continue
            comparable += 1
            if y_true[i] > y_true[j]:
                concordant += (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                concordant += (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
    return concordant / comparable if comparable else np.nan


def regression_sd(y_true, y_pred):
    y_pred_mean = np.mean(y_pred)
    y_true_mean = np.mean(y_true)
    denominator = np.sum((y_pred - y_pred_mean) ** 2)
    if denominator == 0:
        return np.nan
    slope = np.sum((y_pred - y_pred_mean) * (y_true - y_true_mean)) / denominator
    intercept = y_true_mean - slope * y_pred_mean
    y_fit = slope * y_pred + intercept
    return np.sqrt(np.sum((y_true - y_fit) ** 2) / (len(y_pred) - 1))


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "PCC": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "SD": float(regression_sd(y_true, y_pred)),
        "CI": float(concordance_index(y_true, y_pred)),
    }


def predict_dataset(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    records = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for pdb_ids, pocket_seq, smi, bg_seq, mask_pocket, affinity in loader:
            pocket_seq = pocket_seq.to(device)
            smi = smi.to(device)
            bg_seq = bg_seq.to(device)
            mask_pocket = mask_pocket.to(device)
            predictions = model(pocket_seq, smi, bg_seq, smi, mask_pocket)
            predictions = predictions.detach().cpu().numpy().reshape(-1)
            affinity = affinity.detach().cpu().numpy().reshape(-1)
            for pdb_id, true_value, pred_value in zip(pdb_ids, affinity, predictions):
                row = {"pdb_id": pdb_id, "predicted_affinity": float(pred_value)}
                if not np.isnan(true_value):
                    row["experimental_affinity"] = float(true_value)
                    y_true.append(float(true_value))
                    y_pred.append(float(pred_value))
                records.append(row)
    return pd.DataFrame(records), np.asarray(y_true), np.asarray(y_pred)


def dataset_label(path):
    name = Path(path).stem.lower()
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def main(args):
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    metric_rows = []
    for input_path in args.input:
        dataset = FGBindDataset(input_path)
        label = dataset_label(input_path)
        predictions, y_true, y_pred = predict_dataset(
            model, dataset, args.batch_size, device
        )
        prediction_path = output_dir / f"{label}_predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        print(f"Saved predictions: {prediction_path}")

        if len(y_true) > 0:
            metrics = calculate_metrics(y_true, y_pred)
            metric_rows.append({"dataset": label, "n": int(len(y_true)), **metrics})
            print(
                f"{label}: "
                f"RMSE={metrics['RMSE']:.3f}, "
                f"PCC={metrics['PCC']:.3f}, "
                f"MAE={metrics['MAE']:.3f}, "
                f"SD={metrics['SD']:.3f}, "
                f"CI={metrics['CI']:.3f}"
            )

    if metric_rows:
        metrics_path = output_dir / "metrics_summary.csv"
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FGBind-Net predictions on one or more CSV datasets."
    )
    parser.add_argument(
        "--model",
        default="fgbind_net_inference.pt",
        help="TorchScript model path. Default: fgbind_net_inference.pt.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for prediction and metric outputs. Default: results.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
