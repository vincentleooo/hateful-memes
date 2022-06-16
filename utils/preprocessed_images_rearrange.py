import pandas as pd
import numpy as np
import shutil
import os

def rearrange(test: bool):
    if test:
        csv_dir = "preprocessed/test_seen.csv"
        dir = "test_seen"
    else:
        csv_dir = "preprocessed/train.csv"
        dir = "img"
    df = pd.read_csv(csv_dir)
    df["new_dir"] = np.where(df.label == 0, "unhateful/", "hateful/")
    id_str = df["id"].astype("str")
    df["new_dir"] = df["new_dir"] + id_str + ".png"

    if not os.path.isdir(f"preprocessed/{dir}/unhateful"):
        os.makedirs(f"preprocessed/{dir}/unhateful")
    
    if not os.path.isdir(f"preprocessed/{dir}/hateful"):
        os.makedirs(f"preprocessed/{dir}/hateful")

    df.apply(
        lambda row: shutil.move(
            "preprocessed/" + row["img"], f"preprocessed/{dir}/" + row["new_dir"]
        ), axis=1
    )

if __name__ == "__main__":
    raise NotImplementedError
