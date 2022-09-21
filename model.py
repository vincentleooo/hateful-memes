from unicodedata import bidirectional
import torch
from torch import dropout, nn
import pickle

with open("preprocessed/text_object", "rb") as f:
    text_obj = pickle.load(f)
    f.close()


class Model(nn.Module):
    def __init__(self, weights) -> None:
        super().__init__()

        self.drp = 0.1
        self.n_classes = 2

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.drp)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights))

        # Text LSTM
        self.lstm = nn.LSTM(
            300, 30, bidirectional=False, batch_first=True, num_layers=3, dropout=self.drp
        )

        # Text CNN
        self.text_cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(self.drp),
            # Layer 2
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(self.drp),
            # Layer 3
            nn.Conv2d(3, 3, kernel_size=3, stride=(1, 10), padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(self.drp),
        )

        # Image CNN
        self.image_cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 3, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(self.drp),
            # Layer 2
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.Dropout(self.drp),
            # Layer 3
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=0),
            nn.Dropout(self.drp),
        )
    
    def forward(self, x):
        img, text, labels = x
        out_img = self.image_cnn(img)
        print(out_img.shape)
        out_text = self.text_cnn(text)
        print(out_text.shape)

        return out_text


