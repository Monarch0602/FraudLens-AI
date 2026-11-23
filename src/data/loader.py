
from pathlib import Path
from io import StringIO
import pandas as pd


DATA_PATH = Path("data/raw/banksim.csv")


def load_raw_data(path=DATA_PATH):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    print(f"Loading dataset: {path}")

    # utf-8-sig automatically removes the BOM
    with open(path, "r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Remove extra quotation marks surrounding each complete line
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        cleaned_lines.append(line)

    # Parse cleaned CSV
    data = pd.read_csv(
        StringIO("\n".join(cleaned_lines)),
        sep=","
    )

    # The first column has an empty header, so name it step
    first_column = data.columns[0]

    if str(first_column).strip() == "" or str(first_column).startswith("Unnamed"):
        data.rename(
            columns={first_column: "step"},
            inplace=True
        )

    print("Dataset loaded successfully.")
    print(f"Shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print(f"Columns: {list(data.columns)}")

    return data
