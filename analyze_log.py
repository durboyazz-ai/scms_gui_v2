
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Analyze an SCMS session CSV log.")
    ap.add_argument("csv", help="Path to session CSV log produced by SCMS.")
    ap.add_argument("--out", default=None, help="Output folder for plots (default: same folder as CSV).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out) if args.out else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise SystemExit("Invalid log file: missing 'timestamp' column.")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]

    # Summary
    print("\nSession Summary")
    print("-" * 40)
    print(f"File: {csv_path.name}")
    print(f"Frames: {len(df)}")
    if "label" in df.columns:
        print(df["label"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
    if "fatigue" in df.columns:
        fatigue_rate = (df["fatigue"].astype(str).str.lower() == "true").mean() * 100
        print(f"Fatigue frames: {fatigue_rate:.2f}%")

    # Plot concentration
    if "concentration" in df.columns:
        df["concentration"] = pd.to_numeric(df["concentration"], errors="coerce")

        plt.figure(figsize=(12, 4))
        plt.plot(df["t"], df["concentration"])
        plt.xlabel("Time (s)")
        plt.ylabel("Concentration (%)")
        plt.title("SCMS Concentration Over Time")
        plt.tight_layout()
        out1 = out_dir / (csv_path.stem + "_concentration.png")
        plt.savefig(out1, dpi=200)
        plt.close()
        print(f"Saved: {out1}")

    # Plot state (steps)
    if "label" in df.columns:
        mapping = {"Attentive": 2, "Distracted": 1, "Drowsy": 0, "Face Only": -1, "No Face": -2}
        y = df["label"].map(mapping).fillna(-3)

        plt.figure(figsize=(12, 3))
        plt.step(df["t"], y, where="post")
        plt.xlabel("Time (s)")
        plt.ylabel("State (numeric)")
        plt.title("SCMS Attention State Timeline (Encoded)")
        plt.tight_layout()
        out2 = out_dir / (csv_path.stem + "_state.png")
        plt.savefig(out2, dpi=200)
        plt.close()
        print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
