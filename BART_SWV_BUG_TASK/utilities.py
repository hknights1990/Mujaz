from transformers import (
    AdamW,
    BartForConditionalGeneration,
    BartTokenizerFast as BartTokenizer
)
#from pl_data_module import CVESummaryDataModule
#from pytorch_dataset import CVESummaryDataset
#from summarization_model import CVESummaryModel
#from utilities import compression_ratio, data_preprocessing, summarization, summarize
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


def data_preprocessing(training_path, testing_path):

    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(testing_path)

    train_df_swv = train_df[train_df.prefix == 'swv']
    train_df_swv = train_df_swv.rename(
        columns={'input_text': 'text', 'target_text': 'summary'})
    train_df_swv.drop('prefix', axis=1, inplace=True)

    train_df_bug = train_df[train_df.prefix == 'bug']
    train_df_bug = train_df_bug.rename(
        columns={'input_text': 'text', 'target_text': 'summary'})
    train_df_bug.drop('prefix', axis=1, inplace=True)

    test_df = test_df[test_df.prefix == 'sum']
    test_df = test_df.rename(
        columns={'input_text': 'text', 'target_text': 'summary'})

    input_text = test_df['text'].tolist()
    swv_bug_list = [sentence for (sentence) in input_text]
    target_text = test_df['summary'].tolist()

    train_df_swv, test_df_swv = train_test_split(
        train_df_swv, test_size=0.1, random_state=42)

    return train_df_swv, test_df_swv, train_df_bug, input_text, target_text, swv_bug_list


def compression_ratio(original, new):
    return len(new) / len(original)


# summarize function is used in text generation to define a decoding method

# for Top K sampling add parameter top_k
# for Top Nucleus sampling add parameters top_k, top_p


def summarize(text, trained_model, tokenizer):
    trained_model = trained_model
    tokenizer = tokenizer
    text_encoding = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    generated_ids = trained_model.model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=250,
        num_beams=2,
        repetition_penalty=2.0,
        length_penalty=2.0,
        early_stopping=True
    )

    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True,
                         clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]

    return "".join(preds)


def summarizaing_swv(text, target, trained_model, tokenizer, swv_bug_list, concat=False):
    print(60 * "==" + "\n" + 20 * "==" + "Summarizing" + "==" * 20 + "\n" + 60 * "==" + "\n")
    running_sum = 0
    model_swv_summaries, target_summaries = [], []

    for i in range(len(swv_bug_list)):

        swv_sample = swv_bug_list[i]
        model_swv_summary = summarize(swv_sample, trained_model, tokenizer)
        model_swv_summaries.append(model_swv_summary)
        target_sample = target[i]
        target_summaries.append(target_sample)

    return model_swv_summaries, target_summaries


def summarizaing_bug(text, target, trained_model, tokenizer, swv_bug_list, concat=False):
    print(60 * "==" + "\n" + 20 * "==" + "Summarizing" + "==" * 20 + "\n" + 60 * "==" + "\n")

    # trained_model=trained_model
    running_sum = 0
    model_bug_summaries = []

    for i in range(len(swv_bug_list)):

        bug_sample = swv_bug_list[i]
        model_bug_summary = summarize(bug_sample, trained_model, tokenizer)
        model_bug_summaries.append(model_bug_summary)

    return model_bug_summaries
