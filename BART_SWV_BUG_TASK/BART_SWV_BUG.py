# Fine Tuning T5 on the SUM task


from transformers import (
    AdamW,
    BartForConditionalGeneration,
    BartTokenizerFast as BartTokenizer
)
from pl_data_module import CVESummaryDataModule
from pytorch_dataset import CVESummaryDataset
from summarization_model import CVESummaryModel
from utilities import compression_ratio, data_preprocessing, summarizaing_swv, summarizaing_bug, summarize
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


def training(model, data_module, n_EPOCHS):

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        # logger=logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=n_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30
    )

    print(60 * "==" + "\n" + 20 * "==" + "Training Start" +
          "==" * 20 + "\n" + 60 * "==" + "\n")

    trainer.fit(model, data_module)

    print(60 * "==" + "\n" + 20 * "==" + "Finished Training" +
          "==" * 20 + "\n" + 60 * "==" + "\n")
    print(60 * "==" + "\n" + 20 * "==" + "Saving Model..." +
          "==" * 20 + "\n" + 60 * "==" + "\n")

    trained_model = CVESummaryModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    trained_model.freeze()

    return trained_model


def create_model(train_df, test_df):

    MODEL_NAME = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    n_EPOCHS = 4
    BATCH_SIZE = 4

    data_module = CVESummaryDataModule(
        train_df, test_df, tokenizer, batch_size=BATCH_SIZE)
    model = CVESummaryModel()

    trained_model = training(model, data_module, n_EPOCHS)
    return trained_model, tokenizer


def compute_scores(
    model_swv_summaries,
    model_bug_summaries,
    input_text,
    target_summaries
):
    running_sum = 0
    concat_summaries = [
        swv + " " + bug for (swv, bug) in zip(model_swv_summaries, model_bug_summaries)]
    for i in range(len(model_swv_summaries)):
        print(
            f"\nOriginal: {input_text[i]} \nTarget: {target_summaries[i]} \nPredicted: {concat_summaries[i]}")
        running_sum += compression_ratio(input_text[i], concat_summaries[i])
    scores = rouge.get_scores(concat_summaries, target_summaries, avg=True)

    print(f'Rouge: {scores}')
    print(f'Avg Compression: {running_sum / len(input_text)}')


def main():
    print(20 * "==" + "Starting main" + "==" * 20)

    TRAIN_CSV_PATH = '../Datasets/dataset_multi_train.csv'
    TEST_CSV_PATH = '../Datasets/dataset_multi_test.csv'

    train_df_swv, test_df_swv, train_df_bug, input_text, target_text, swv_bug_list = data_preprocessing(
        TRAIN_CSV_PATH, TEST_CSV_PATH)
    print("Dataset has been loaded. size of training " + str(train_df_swv.shape))

    trained_model_swv, tokenizer = create_model(
        train_df_swv,
        test_df_swv
    )

    model_swv_summaries, target_summaries = summarizaing_swv(
        input_text,
        target_text,
        trained_model_swv,
        tokenizer,
        swv_bug_list
    )

    train_df_bug, test_df_bug = train_test_split(
        train_df_bug, test_size=0.1, random_state=42)

    trained_model_bug, tokenizer = create_model(
        train_df_bug,
        test_df_bug
    )

    model_bug_summaries = summarizaing_bug(
        input_text,
        target_text,
        trained_model_bug,
        tokenizer,
        swv_bug_list
    )
    compute_scores(
        model_swv_summaries,
        model_bug_summaries,
        input_text,
        target_summaries
    )


if __name__ == '__main__':
    main()
