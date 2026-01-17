import torch
import pandas as pd
from typing import List, Dict, Set
from gensim.models import KeyedVectors


def load_sst2_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep="\t", header=0)


def get_vocabulary(df: pd.DataFrame) -> Set[str]:
    vocabulary = set()
    for text in df["sentence"]:
        vocabulary.update(text.lower().split())
    return vocabulary


def load_word_embeddings(model_path: str, vocabulary: Set[str]) -> Dict[str, int]:
    word_to_id = {"<PAD>": 0}
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    for word in vocabulary:
        if word in model.key_to_index:
            word_to_id[word] = len(word_to_id)

    return word_to_id


def convert_text_to_ids(text: str, word_to_id: Dict[str, int]) -> List[int]:
    tokens = text.lower().split()
    ids = [word_to_id[token] for token in tokens if token in word_to_id]

    return ids


def process_sst2_data(file_path: str, word_to_id: Dict[str, int]) -> List[Dict]:
    df = load_sst2_data(file_path)

    processed_data = []

    for _, row in df.iterrows():
        input_ids = convert_text_to_ids(row["sentence"], word_to_id)
        if not input_ids:
            continue
        data = {
            "text": row["sentence"],
            "label": torch.tensor([float(row["label"])]),
            "input_ids": torch.tensor(input_ids),
        }
        processed_data.append(data)

    return processed_data


def main():
    train_df = load_sst2_data("../Section7/SST-2/train.tsv")
    dev_df = load_sst2_data("../Section7/SST-2/dev.tsv")
    vocabulary = get_vocabulary(train_df)
    vocabulary.update(get_vocabulary(dev_df))
    word_to_id = load_word_embeddings(
        "../Section6/GoogleNews-vectors-negative300.bin", vocabulary
    )
    train_data = process_sst2_data("../Section7/SST-2/train.tsv", word_to_id)
    print(f"訓練データ数: {len(train_data)}")
    dev_data = process_sst2_data("../Section7/SST-2/dev.tsv", word_to_id)
    print(f"開発データ数: {len(dev_data)}")
    print("\n訓練データのサンプル:")
    sample = train_data[0]
    print(f"テキスト: {sample['text']}")
    print(f"ラベル: {sample['label']}")
    print(f"トークンID列: {sample['input_ids']}")


if __name__ == "__main__":
    main()