import os
import pandas as pd

def update_labels_in_csv(directory_path):
    """
    Reads all CSV files in the given directory, replaces 'NeedManualLabel' in the 'Label' column
    with the label derived from filename (substring before first '_'), and overwrites the file.
    Returns a list of processed file names.
    """

    processed_files = []

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(directory_path, filename)

            try:
                # Derive label from filename (part before first "_")
                label_name = filename.split("_")[0]

                # Read CSV using pandas
                df = pd.read_csv(file_path)

                # Check if 'Label' column exists
                if 'Label' not in df.columns:
                    print(f"⚠️ Skipping {filename}: No 'Label' column found.")
                    continue

                # Replace 'NeedManualLabel' with derived label
                df['Label'] = df['Label'].replace('NeedManualLabel', label_name)

                # Overwrite the same file
                df.to_csv(file_path, index=False)

                processed_files.append(filename)
                print(f"✅ Updated: {filename} → Label set to '{label_name}'")

            except Exception as e:
                print(f"X Error processing {filename}: {e}")

    print(f"\n🎯 Done! {len(processed_files)} files updated.")
    return processed_files


if __name__ == "__main__":
    # Example usage — replace this path with your actual folder
    directory = r"H:\Datasets\CIC-IoT-IDAD-2024\Flow_Based"

    updated_files = update_labels_in_csv(directory)

    print("\n📄 Files processed:")
    for f in updated_files:
        print(" -", f)
