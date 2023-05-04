# Creating Dataset to be used by  pytorch Dataloader

from transformers import (
    AdamW,
    BartForConditionalGeneration,
    BartTokenizerFast as BartTokenizer
)
#from pl_data_module import CVESummaryDataModule
#from pytorch_dataset import CVESummaryDataset
from summarization_model import CVESummaryModel
from utilities import compression_ratio, data_preprocessing, summarizaing_swv, summarizaing_bug
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from rouge import Rouge
rouge = Rouge()

# custom imports
#from utilities import compression_ratio, summarize, data_preprocessing, summarization
#from summarization_model import CVESummaryModel
#from pytorch_dataset import CVESummaryDataset
#from pl_data_module import CVESummaryDataModule


class CVESummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: BartTokenizer,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128
    ):

        self.data = data
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row["text"]

        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        summary_encoding = self.tokenizer(
            data_row["summary"],
            max_length=self.summary_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            text=text,
            summary=data_row["summary"],
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten()
        )
