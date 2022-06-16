from mmdet.apis import inference_detector
from mmdet.core import get_classes

import numpy as np
import pandas as pd

from tqdm import tqdm

from PIL import Image
from utils.mmdet_model_loader import init
from utils.preprocessed_images_rearrange import rearrange

import os
import sys
import logging
import warnings
import argparse

warnings.filterwarnings("ignore")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser(description="Image preprocessor")
    parser.add_argument(
        "--test",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Runs on the seen test dataset.",
    )

    opt = parser.parse_args()

    out_dir = "preprocessed/img"

    if opt.test:
        json_dir = "./hateful_memes/test_seen.jsonl"
    else:
        json_dir = "./hateful_memes/train.jsonl"

    logging.info("Starting image preprocessing...")
    if not os.path.isdir(out_dir):
        logging.warning("Preprocessed folder does not exist. Creating one...")
        os.mkdir(out_dir)
    df = pd.read_json(json_dir, lines=True)
    classes = get_classes("coco")
    labels_classes = []
    model = init()

    for i in tqdm(df.img):
        img_path = f"./hateful_memes/{i}"
        result = inference_detector(model, img_path)
        bbox_result = result
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)
        labels_impt = np.where(bboxes[:, -1] > 0.3)[0]
        labels_impt_list = [labels[i] for i in labels_impt]
        labels_class = [classes[i] for i in labels_impt_list]
        labels_classes.append(labels_class)

        impt_bboxes = bboxes[labels_impt]

        try:
            max_box_array = impt_bboxes.max(axis=0)
            min_box_array = impt_bboxes.min(axis=0)

            # Opens a image in RGB mode
            im = Image.open(img_path)

            # Setting the points for cropped image
            left = min_box_array[0]
            top = min_box_array[1]
            right = max_box_array[2]
            bottom = max_box_array[3]

            # Cropped image of above dimension
            # (It will not change original image)
            im1 = im.crop((left, top, right, bottom))
            im1 = im1.save(f"./preprocessed/{i}")
        except:
            im = Image.open(img_path)
            im = im.save(f"./preprocessed/{i}")

    df["labels"] = labels_classes

    logging.info("Dumping CSV file...")

    if opt.test:
        df.to_csv("./preprocessed/test_seen.csv")
    else:
        df.to_csv("./preprocessed/train.csv")

    logging.info("Rearranging")
    rearrange(opt.test)

    logging.info("Done.")


if __name__ == "__main__":
    main()
