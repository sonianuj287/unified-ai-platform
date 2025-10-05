# src/preprocess.py
import argparse
import pandas as pd
import numpy as np
import os

def preprocess(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path)

    # If there's a timestamp column, sort by it and drop after
    for col in df.columns:
        if "time" in col.lower() or "timestamp" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col)
                df = df.drop(columns=[col])
                break
            except Exception:
                pass

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        raise ValueError("No numeric columns found after preprocessing")

    # Save processed CSV
    df.to_csv(output_path, index=False)
    print(f"[preprocess] saved processed data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output)
