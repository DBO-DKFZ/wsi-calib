from typing import Union
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold


def substract_test_slides(slide_info: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    # Code from HistoSlides._adjust_df()
    assert "slide_id" in test_df.columns
    test_ids = test_df["slide_id"].to_list()
    train_df = slide_info[~slide_info["slide_id"].isin(test_ids)].reset_index(drop=False)  # Use tilde to negate isin()
    return train_df


def replace_column_names(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    old_columns = df.columns
    new_columns = old_columns.map(mapping)
    df.columns = new_columns
    return df


def main(
    root_p: Union[str, Path],
    n_folds: int,
    seed: int,
    stratified: bool,
):
    root_p = Path(root_p)
    slide_info_p = root_p / "slide_information.csv"
    assert slide_info_p.exists()
    test_slides_p = root_p / "test_slides.csv"
    assert test_slides_p.exists()

    slide_info = pd.read_csv(slide_info_p)
    slide_info = replace_column_names(df=slide_info, mapping={"Pseudonym": "slide_id", "Label": "label"})
    # print(slide_info.head())
    test_df = pd.read_csv(test_slides_p)

    train_df = substract_test_slides(slide_info, test_df)

    out_p = root_p / "folds"
    out_p.mkdir(exist_ok=True)

    # Important to use the same seed for all folds
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        all_splits = [k for k in kf.split(X=train_df, y=train_df["label"])]
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        all_splits = [k for k in kf.split(X=train_df, y=None)]
    for idx in range(len(all_splits)):
        train_indices, val_indices = all_splits[idx]  # index 0 - 1-n_splits  # Returns two numpy arrays
        train_indices, val_indices = train_indices.tolist(), val_indices.tolist()
        # Write dfs for folds into csv files
        train_df_out = train_df.loc[train_indices]
        val_df_out = train_df.loc[val_indices]

        print(f"Writing files for fold {idx + 1}")
        train_df_out.to_csv(str(out_p / f"fold_{idx + 1}_train.csv"), index=False)
        val_df_out.to_csv(str(out_p / f"fold_{idx + 1}_val.csv"), index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--root_p", type=str, required=True, help="Path to where dataset information is stored")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified", type=bool, default=True)

    args = parser.parse_args()
    cfg = vars(args)

    main(**cfg)
