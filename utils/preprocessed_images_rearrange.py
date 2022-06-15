import pandas as pd
import numpy as np
import shutil
import os

def rearrange():
    df = pd.read_csv("preprocessed/train.csv")
    df["new_dir"] = np.where(df.label == 0, "unhateful/", "hateful/")
    id_str = df["id"].astype("str")
    df["new_dir"] = df["new_dir"] + id_str + ".png"

    if not os.path.isdir("preprocessed/img/unhateful"):
        os.mkdir("preprocessed/img/unhateful")
    
    if not os.path.isdir("preprocessed/img/hateful"):
        os.mkdir("preprocessed/img/hateful")

    df.apply(
        lambda row: shutil.move(
            "preprocessed/" + row["img"], "preprocessed/img/" + row["new_dir"]
        ), axis=1
    )

if __name__ == "__main__":
    raise NotImplementedError
