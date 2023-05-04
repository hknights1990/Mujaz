
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


def data_preprocessing(training_path, testing_path):
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(testing_path)

    # Drop all non-summary

    train_df = train_df[train_df.prefix == 'sum']
    train_df.drop('prefix', axis=1, inplace=True)

    test_df = test_df[test_df.prefix == 'sum']
    test_df.drop('prefix', axis=1, inplace=True)

    # Rename columns
    train_df = train_df.rename(
        columns={'input_text': 'text', 'target_text': 'summary'})
    test_df = test_df.rename(
        columns={'input_text': 'text', 'target_text': 'summary'})

    input_text = test_df['text'].tolist()
    target_text = test_df['summary'].tolist()

    train_df, test_df = train_test_split(
        train_df, test_size=0.1, random_state=42)

    return train_df, test_df, input_text, target_text


def compression_ratio(original, new):
    return len(new) / len(original)


# Summarize function is used in text generation to define a decoding method

# For Top K sampling add parameter top_k
# For Top Nucleus sampling add parameters top_k, top_p

def summarize(text, trained_model, tokenizer):

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


def summarization(text, target, trained_model, tokenizer):
    print(60 * "==")
    print(25 * "==" + "Summarizing....." + "==" * 25)
    print(60 * "==")

    running_sum = 0
    model_summaries, target_summaries = [], []

    for i in range(len(text)):

        sum_text = text[i]
        model_summary = summarize(sum_text, trained_model, tokenizer)
        target_summary = target[i]

        model_summaries.append(model_summary)
        target_summaries.append(target_summary)
        running_sum += compression_ratio(sum_text, model_summary)

        print(
            f"\nOriginal: {sum_text} \nTarget: {target_summary} \nPredicted: {model_summary}")

    scores = rouge.get_scores(model_summaries, target_summaries, avg=True)

    return scores, running_sum
