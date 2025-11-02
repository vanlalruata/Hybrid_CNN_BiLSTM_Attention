import os
import csv


def convert_csv_files(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each CSV file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, 'r', encoding='utf-8', newline='') as infile:
                # Use csv.reader to handle quoted and semicolon-delimited data
                reader = csv.reader(infile, delimiter=';', quotechar='"')

                # Write as comma-delimited, without quotes
                with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
                    writer = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

                    for row in reader:
                        # Strip leading/trailing whitespace or empty quotes
                        clean_row = [cell.strip('"').strip() for cell in row]
                        writer.writerow(clean_row)

            print(f"✅ Converted: {filename}")

    print("\n🎉 Conversion complete! Files saved to:", output_dir)


if __name__ == "__main__":
    # Example usage — you can modify these paths
    input_directory = r"H:\Datasets\BoT-IoT"
    output_directory = r"H:\Datasets\BoT-IoT_csvs"

    convert_csv_files(input_directory, output_directory)
