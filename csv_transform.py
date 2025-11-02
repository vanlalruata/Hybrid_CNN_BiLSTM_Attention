import os
import re
import csv
from io import StringIO
import pandas as pd

LABEL_PREFIX_RE = re.compile(r'(?i)^PartOfA(?:Horizontal|Vertical)?')

def clean_label(value: str) -> str:
    if not isinstance(value, str):
        return value
    # Strip outer quotes/spaces then remove the prefix at the START only
    v = value.strip().strip('"').strip("'")
    v = LABEL_PREFIX_RE.sub('', v)
    return v.strip()

def normalize_header_and_read(path: str) -> pd.DataFrame:
    """
    Reads a pipe-delimited file, fixing the common case where the HEADER line starts with '|'
    (creating an empty first column). Ensures correct column parsing.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()

    if not lines:
        return pd.DataFrame()

    # If header starts with a pipe, remove just that leading pipe on the header line
    header = lines[0]
    if header.lstrip().startswith('|'):
        # remove the very first '|' (not every pipe)
        first_pipe_idx = header.find('|')
        if first_pipe_idx == 0:
            header = header[1:]
        else:
            # lstrip caused offset; rebuild safely
            prefix_len = len(lines[0]) - len(lines[0].lstrip())
            header = lines[0][:prefix_len] + lines[0][prefix_len+1:]

        lines[0] = header

    # Some sources accidentally quote entire rows; strip one pair of wrapping quotes if present
    def strip_wrapping_quotes(s: str) -> str:
        s_stripped = s.strip()
        if len(s_stripped) >= 2 and s_stripped[0] == '"' and s_stripped[-1] == '"':
            return s_stripped[1:-1]
        return s

    # Only apply to clearly broken cases (whole-row quoted). Safe for header + data.
    lines = [strip_wrapping_quotes(ln) for ln in lines]

    # Rejoin to a buffer and parse as pipe-delimited
    buffer = StringIO('\n'.join(lines))
    df = pd.read_csv(buffer, sep='|', engine='python', dtype=str)

    # If there is an empty first column due to odd formatting, drop it
    if '' in df.columns:
        df = df.drop(columns=[''])

    return df

def find_label_column(df: pd.DataFrame) -> str | None:
    # Support both 'label' and 'Label' (and similar case variants)
    for col in df.columns:
        if col.lower() == 'label':
            return col
    return None

def convert_pipe_to_comma(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.csv'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            df = normalize_header_and_read(input_path)

            # Clean the label column if present
            label_col = find_label_column(df)
            if label_col is not None:
                df[label_col] = df[label_col].apply(clean_label)

            # Save as comma-delimited; QUOTE_MINIMAL avoids whole-line quoting
            df.to_csv(
                output_path,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
                quotechar='"',
            )
            print(f"✅ Converted and cleaned: {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print("\n🎉 All files processed successfully (comma-delimited, no row-wide quotes).")

def main():
    print("=== Pipe (|) → Comma (,) CSV Converter with Label Cleaner ===")
    input_dir = r"H:\Datasets\Aposemat-IoT-23\aposemat_iot_23\combined"
    output_dir = r"H:\Datasets\Aposemat-IoT-23\aposemat_iot_23\processed"

    if not os.path.exists(input_dir):
        print("❌ Error: Input directory does not exist.")
        return

    convert_pipe_to_comma(input_dir, output_dir)

if __name__ == "__main__":
    main()
