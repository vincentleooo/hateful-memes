from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import contractions
import nltk
import gensim.downloader as api
import logging
import sys
import pickle
import os
import ast
import re
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

class CorpusToEmbedding:
    def __init__(self, text_series: pd.Series, labels: pd.Series, max_length: int, labels_length: int):
        self._text = text_series
        self._tokenizer = RegexpTokenizer("[a-zA-Z0-9]\w+")
        self._lemmatizer = WordNetLemmatizer()
        logging.info("Loading pretrained weights. This might take a while.")
        self._pretrained_weights = api.load("fasttext-wiki-news-subwords-300")
        self._tokens = None
        self._labels = labels
        self.max_length = max_length
        self.labels_length = labels_length

        # for padding purposes
        self._weights = np.append(
            self._pretrained_weights.vectors, np.zeros((1, 300)), axis=0
        )

        self._preprocess()
        self._text = self._text.apply(self._parse_list)
        self._text = self._text.apply(self._padding, args=(self.max_length,))
        self._labels = self._labels.apply(self._parse_list, args=(True,))
        self._labels = self._labels.apply(self._padding, args=(self.labels_length,))
        self._tokens = self._text

    @property
    def text(self):
        return self._text

    @property
    def weights(self):
        return self._weights

    @property
    def tokens(self):
        return self._tokens

    @property
    def labels(self):
        return self._labels

    @staticmethod
    def nltk_downloads():
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    def _preprocess(self):
        self._text = self._text.apply(contractions.fix)
        self._text = self._text.apply(self._tokenizer.tokenize)
        self._text = self._text.apply(
            lambda text: [self._lemmatizer.lemmatize(word) for word in text]
        )

    def _get_index(self, x):
        try:
            index = self._pretrained_weights.get_index(x)
            return index
        except:
            logging.warn(f"{x} is not indexable.")

    def _parse_list(self, ls: list, labels=False):
        ret_ls = []
        for i in ls:
            if labels:
                i = re.sub(r".+?_", "", i)
            x = self._get_index(i)
            if x is not None:
                ret_ls.append(x)

        return ret_ls

    def _padding(self, ls, max_length):
        while len(ls) < max_length:
            ls.append(999999)
        if len(ls) > max_length:
            ls = ls[0 : max_length]
        return ls


def main():
    parser = argparse.ArgumentParser(description="Image preprocessor")
    parser.add_argument(
        "--test",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Runs on the seen test dataset.",
    )

    opt = parser.parse_args()
    
    if opt.test:
        dir = "test_seen"
    else:
        dir = "train"

    logging.info("You have accessed this file as a script.")
    df = pd.read_csv(f"preprocessed/{dir}.csv")
    df["labels"] = df.labels.apply(ast.literal_eval)
    text_object = CorpusToEmbedding(df.text, df.labels, 30, 3)

    if not os.path.isdir("preprocessed"):
        logging.warning("Preprocessed folder does not exist. Creating one...")
        os.mkdir("preprocessed")
    logging.info(f"Dumping into pickle file...")
    outfile = open(f"preprocessed/text_object_{dir}", "wb")
    pickle.dump(text_object, outfile)
    outfile.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()