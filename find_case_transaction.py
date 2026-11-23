
import pandas as pd

def main():
    case_step = 158291
    case_amount = 236.14739999999958
    case_category = "es_leisure"
    case_age = "2"
    case_gender = "M"

    file_path = "data/raw/banksim.csv"

    # Read the file line-by-line to handle its unusual quotation format
    rows = []

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Skip the malformed header
    for line in lines[1:]:
        line = line.strip()

        # Remove surrounding quotes from each row
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        values = line.split(",")

        if len(values) == 6:
            rows.append(values)
        else:
            print(f"Skipping malformed row: {line}")

    columns = [
        "step",
        "age",
        "gender",
        "category",
        "amount",
        "fraud"
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Convert numeric columns
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["fraud"] = pd.to_numeric(df["fraud"], errors="coerce")

    print("=" * 60)
    print("FRAUDLENS AI - SOURCE TRANSACTION MATCH")
    print("=" * 60)

    print("\nDataset columns:")
    print(df.columns.tolist())

    print(f"\nTotal valid rows: {len(df)}")

    # Match case to source transaction
    matches = df[
        (df["step"] == case_step) &
        (df["category"] == case_category) &
        ((df["amount"] - case_amount).abs() < 0.0001) &
        (df["age"].astype(str) == case_age) &
        (df["gender"] == case_gender)
    ].copy()

    print("\nMatching transactions:")
    print(matches.to_string(index=False))

    print(f"\nNumber of matches: {len(matches)}")

    if len(matches) == 1:
        print("\nSUCCESS: Exact transaction match found.")
        print("\nSource transaction details:")
        print(matches.iloc[0].to_dict())

    elif len(matches) == 0:
        print("\nNo exact match found.")
        print("The case may have been generated from a different")
        print("dataset mapping or preprocessing pipeline.")

    else:
        print("\nMultiple matches found.")
        print("We need to identify the correct source row.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
