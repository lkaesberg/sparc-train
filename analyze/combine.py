import pandas as pd
import glob
import os

def combine_stats(folder="."):
    # Find all *_stats.csv files
    files = glob.glob(os.path.join(folder, "*_stats.csv"))
    if not files:
        print("No *_stats.csv files found in the folder.")
        return

    combined = pd.DataFrame()

    for file in files:
        # Extract model name from filename (before "_stats.csv")
        model_name = os.path.basename(file).replace("_stats.csv", "")

        # Read file
        df = pd.read_csv(file)

        # Drop completely empty rows
        df = df.dropna(how="all")

        # Keep only Metric and Value
        df = df[["Metric", "Value"]].dropna()

        # Add model column
        df = df.rename(columns={"Value": model_name})

        if combined.empty:
            combined = df
        else:
            combined = pd.merge(combined, df, on="Metric", how="outer")

    # Save combined table
    output_file = os.path.join(folder, "combined_stats.csv")
    combined.to_csv(output_file, index=False)
    print(f"✅ Combined stats saved to {output_file}")

if __name__ == "__main__":
    combine_stats(".")

