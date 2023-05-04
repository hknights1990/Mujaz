# Fine Tuning T5 on the SUM task


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from pl_data_module import CVESummaryDataModule
from pytorch_dataset import CVESummaryDataset
from summarization_model import CVESummaryModel
from utilities import compression_ratio, data_preprocessing, summarize, summarization
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


def main():
    print(20 * "==" + "starting main" + "==" * 20)

    TRAIN_CSV_PATH = '../Datasets/dataset_multi_train.csv'
    TEST_CSV_PATH = '../Datasets/dataset_multi_test.csv'

    train_df, test_df, input_text, target_text = data_preprocessing(
        TRAIN_CSV_PATH, TEST_CSV_PATH)
    print("Dataset has been loaded. size of training " + str(train_df.shape))

    MODEL_NAME = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    n_EPOCHS = 4
    BATCH_SIZE = 4

    data_module = CVESummaryDataModule(
        train_df, test_df, tokenizer, batch_size=BATCH_SIZE)

    model = CVESummaryModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    #logger = TensorBoardLogger("lightning_logs", name="Sum Task T5-Base Model")

    trainer = pl.Trainer(
        # logger=logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=n_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30
    )
    print(60 * "==")
    print(20 * "==" + "Training Start" + "==" * 20)
    print(60 * "==")
    trainer.fit(model, data_module)
    print(60 * "==")
    print(20 * "==" + "Finish Training" + "==" * 20)
    print(60 * "==")
    print(20 * "==" + "Saving....." + "==" * 20)
    print(60 * "==")
    trained_model = CVESummaryModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    trained_model.freeze()

    rouge_score, compression = summarization(
        input_text, target_text, trained_model, tokenizer)

    print(f'Rouge: {rouge_score}')
    print(f'Avg Compression: {compression / len(input_text)}')


if __name__ == '__main__':
    main()
