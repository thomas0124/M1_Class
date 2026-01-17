import pandas as pd
from transformers import AutoTokenizer

def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df["sentence"].tolist(), df["label"].tolist()

def tokenize_texts(texts):
    tokenized_texts = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        tokenized_texts.append(tokens)
    return tokenized_texts

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

train_path = "../Section7/SST-2/train.tsv"
dev_path = "../Section7/SST-2/dev.tsv"

train_texts, train_labels = load_data(train_path)
dev_texts, dev_labels = load_data(dev_path)

train_tokenized = tokenize_texts(train_texts)
dev_tokenized = tokenize_texts(dev_texts)