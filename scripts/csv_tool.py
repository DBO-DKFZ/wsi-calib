import os
from pathlib import Path
import pandas as pd


def extract_rows(csv_p: Path, frac: float = 0.2) -> pd.DataFrame:
    df = pd.read_csv(csv_p)
    print(len(df))
    print(df.head())
    df_frac = df.sample(frac=frac, random_state=42)
    df_frac = df_frac.reset_index(drop=True)
    print(len(df_frac))
    print(df_frac.head())
    print(df_frac["Label"].value_counts())
    return df_frac


def store_df(df: pd.DataFrame, out_p: Path):
    old_columns = df.columns
    name_dict = {
        "Pseudonym": "slide_id",
        "Label": "label",
    }
    new_columns = old_columns.map(name_dict)
    df.columns = new_columns
    print(df.head())
    df.to_csv(out_p, index=False)


if __name__ == "__main__":
    csv_p = Path("/mnt/ssd/Data/MCO-SCalib/slide_information.csv")
    out_p = Path("/mnt/ssd/Data/MCO-SCalib/test_slides.csv")
    df = extract_rows(csv_p, frac=0.2)
    store_df(df, out_p)
