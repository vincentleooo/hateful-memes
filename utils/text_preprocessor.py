from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import contractions
import nltk
import gensim.downloader as api
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

class CorpusToEmbedding:
    def __init__(self, text_series: pd.Series, max_length: int):
        self._text = text_series
        self._tokenizer = RegexpTokenizer("[a-zA-Z0-9]\w+")
        self._lemmatizer = WordNetLemmatizer()
        logging.info("Loading pretrained weights. This might take a while.")
        self._pretrained_weights = api.load("fasttext-wiki-news-subwords-300")
        self._tokens = None
        self.max_length = max_length

        # for padding purposes
        self._weights = np.append(
            self._pretrained_weights.vectors, np.zeros((1, 300)), axis=0
        )

        self._preprocess()
        self._text = self._text.apply(self._parse_list)
        self._text = self._text.apply(self._padding)
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

    def _parse_list(self, ls):
        ret_ls = []
        for i in ls:
            x = self._get_index(i)
            if x is not None:
                ret_ls.append(x)

        return ret_ls

    def _padding(self, ls):
        while len(ls) < self.max_length:
            ls.append(999999)
        if len(ls) > self.max_length:
            ls = ls[0 : self.max_length]
        return ls


def main():
    logging.info("You have accessed this file as a script. This will run test mode.")
    df = pd.read_csv("preprocessed/train.csv")
    text_object = CorpusToEmbedding(df.text, 30)
    logging.info(f"\nTokens\n------\n{text_object.tokens}")
    logging.info(f"\nWeights\n-------\n{text_object.weights}")
    logging.info("Done.")


if __name__ == "__main__":
    main()