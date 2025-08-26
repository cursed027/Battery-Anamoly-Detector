import os
import pandas as pd
from sklearn.model_selection import train_test_split


def process_metadata(metadata_path: str, data_folder: str, output_folder: str):
    """Process metadata and save per-battery CSVs."""
    os.makedirs(output_folder, exist_ok=True)

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Filter only discharge cycles
    df = metadata[metadata['type'] == 'discharge']
    df = df.drop(columns=['start_time', 'ambient_temperature', 'Capacity', 'Re', 'Rct'], errors="ignore")
    df['cycle_count'] = df.groupby('battery_id').cumcount() + 1
    df = df.drop(columns=['test_id', 'uid'], errors="ignore")

    # Process per battery
    for battery_id, group in df.groupby("battery_id"):
        merged_df = []

        # Sort by cycle count to keep temporal order
        group = group.sort_values("cycle_count")

        for _, row in group.iterrows():
            file_path = os.path.join(data_folder, row["filename"])

            if os.path.exists(file_path):
                cycle_df = pd.read_csv(file_path)
                cycle_df["battery_id"] = battery_id
                cycle_df["cycle_count"] = row["cycle_count"]
                merged_df.append(cycle_df)
            else:
                print(f"Warning: {file_path} not found, skipping.")

        if merged_df:
            final_df = pd.concat(merged_df, ignore_index=True)
            output_file = os.path.join(output_folder, f"battery_{battery_id}.csv")
            final_df.to_csv(output_file, index=False)
            print(f"Saved {output_file} with {len(final_df)} rows.")


def split_datasets(data_dir: str, output_dir: str, test_size=0.30, val_size=0.50, random_state=42):
    """Split per-battery CSVs into train/val/test datasets and save combined CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    # List all battery files
    all_batteries = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    all_battery_ids = [os.path.splitext(f)[0] for f in all_batteries]

    # Train/Val/Test split
    train_ids, temp_ids = train_test_split(all_battery_ids, test_size=test_size, random_state=random_state)
    val_ids, test_ids = train_test_split(temp_ids, test_size=val_size, random_state=random_state)

    print(f"Train batteries: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # Function to combine cycles
    def combine_batteries(battery_ids, save_path):
        dfs = []
        for b in battery_ids:
            file_path = os.path.join(data_dir, f"{b}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df["battery_id"] = b
                dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(save_path, index=False)
        return final_df

    # Save datasets
    train_df = combine_batteries(train_ids, os.path.join(output_dir, "train_dataset.csv"))
    val_df   = combine_batteries(val_ids, os.path.join(output_dir, "val_dataset.csv"))
    test_df  = combine_batteries(test_ids, os.path.join(output_dir, "test_dataset.csv"))

    print("Final dataset shapes:")
    print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")


if __name__ == "__main__":
    metadata_path = r"datasets\raw\metadata.csv" #give the 
    data_folder = r"datasets\data"      # raw cycles
    battery_output_folder = r"datasets\interim"  # per-battery CSVs
    dataset_output_folder = r"datasets\processed"  # final train/val/test CSVs

    process_metadata(metadata_path, data_folder, battery_output_folder)
    split_datasets(battery_output_folder, dataset_output_folder)
