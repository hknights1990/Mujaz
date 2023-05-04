# 3 Task Concatenation Model 
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
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)

def data_preprocessing(training_path, testing_path):
    
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(testing_path)

    train_df = train_df.rename(columns= {'input_text':'text', 'target_text':'summary'})

    train_df.loc[train_df["prefix"] == "swv", "text"] = "swv: " + train_df["text"]
    train_df.loc[train_df["prefix"] == "bug", "text"] = "bug: " + train_df["text"]
    train_df.loc[train_df["prefix"] == "sum", "text"] = "sum: " + train_df["text"]

    train_df.drop('prefix', axis=1, inplace=True)

    test_df = test_df.rename(columns= {'input_text':'text', 'target_text':'summary'})
    test_df = test_df[test_df.prefix == 'sum']

    # used for testing the model 
    input_text = test_df['text'].tolist()
    target_text = test_df['summary'].tolist()

    swv_list = ['swv: ' + sentence for (sentence) in input_text]
    bug_list = ['bug: ' + sentence for (sentence) in input_text]


    train_df, test_df = train_test_split(train_df, test_size=0.1, random_state=42)

    return train_df, test_df, input_text, target_text, swv_list, bug_list

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


def summarization(text, target, trained_model, tokenizer, swv_list, bug_list):
    print(60 * "==")
    print(25 * "==" + "Summarizing....." + "==" * 25)
    print(60 * "==")

    #trained_model=trained_model
    running_sum = 0
    model_swv_summaries, target_summaries, model_bug_summaries = [],[],[]

    for i in range(len(swv_list)):

        swv_sample = swv_list[i]
        bug_sample = bug_list[i]

        model_swv_summary = summarize(swv_sample, trained_model, tokenizer)
        model_bug_summary = summarize(bug_sample, trained_model, tokenizer)

        model_swv_summaries.append(model_swv_summary)
        model_bug_summaries.append(model_bug_summary)

        target_sample = target[i]
        target_summaries.append(target_sample)

    concat_summaries = [swv + " " + bug for (swv, bug) in zip(model_swv_summaries, model_bug_summaries )]

    for i in range(len(swv_list)):    
        print(f"\nOriginal: {text[i]} \nTarget: {target_summaries[i]} \nPredicted: {concat_summaries[i]}")
        running_sum += compression_ratio(text[i], concat_summaries[i])
    
    scores = rouge.get_scores(concat_summaries, target_summaries, avg=True)
    return scores, running_sum
