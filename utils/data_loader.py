# utils/data_loader.py
from pathlib import Path
import pandas as pd

def get_project_data_dir() -> Path:
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent
    data_dir = project_root / "data" / "prices"
    return data_dir

def load_price_panel() -> pd.DataFrame:
    DATA_DIR = get_project_data_dir()

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Price data folder not found: {DATA_DIR}\n"
            "Make sure your CSVs are in: <project_root>/data/prices/ and you run Streamlit from project root."
        )

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}. "
            "Check that files are named like AAPL.csv, MSFT.csv, etc."
        )

    all_frames = []
    seen_files = []
    for csv_path in csv_files:
        seen_files.append(str(csv_path.name))
        try:
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(csv_path, engine="python", error_bad_lines=False)

        df.columns = [c.lower().strip() for c in df.columns]

        required = ["date", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required):
            raise ValueError(
                f"CSV '{csv_path.name}' is missing required columns.\n"
                f"Found columns: {list(df.columns)}\n"
                f"Required (case-insensitive): {required}"
            )

        df_clean = df[required].copy()
        df_clean["adj_close"] = df_clean["close"]

        try:
            df_clean["date"] = pd.to_datetime(df_clean["date"], infer_datetime_format=True, dayfirst=False)
        except Exception:
            df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")

        df_clean["ticker"] = csv_path.stem.upper()
        all_frames.append(df_clean)

    price_df = pd.concat(all_frames, ignore_index=True)
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    price_df.attrs["source_files"] = seen_files
    return price_df
