import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Set
from gensim.models import KeyedVectors


def load_sst2_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep="\t", header=0)


def get_vocabulary(df: pd.DataFrame) -> Set[str]:
    vocabulary = set()
    for text in df["sentence"]:
        vocabulary.update(text.lower().split())
    return vocabulary


class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(
            embedding_dim, 1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))


def load_word_embeddings(
    model_path: str, vocabulary: Set[str]
) -> tuple[Dict[str, int], torch.Tensor]:
    word_to_id = {"<PAD>": 0}
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    embeddings = [torch.zeros(model.vector_size)]
    for word in vocabulary:
        if word in model.key_to_index:
            word_to_id[word] = len(word_to_id)
            embeddings.append(torch.tensor(model[word]))
    embedding_matrix = torch.stack(embeddings)

    return word_to_id, embedding_matrix


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


def create_mean_embedding_features(
    data: List[Dict], embedding_matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []

    for item in data:
        input_ids = item["input_ids"]
        embeddings = embedding_matrix[input_ids]
        mean_embedding = torch.mean(embeddings, dim=0)

        features.append(mean_embedding)
        labels.append(item["label"])

    return torch.stack(features), torch.cat(labels)


def main():
    train_df = load_sst2_data("../Section7/SST-2/train.tsv")
    dev_df = load_sst2_data("../Section7/SST-2/dev.tsv")
    vocabulary = get_vocabulary(train_df)
    vocabulary.update(get_vocabulary(dev_df))
    word_to_id, embedding_matrix = load_word_embeddings(
        "../Section6/GoogleNews-vectors-negative300.bin", vocabulary
    )
    train_data = process_sst2_data("../Section7/SST-2/train.tsv", word_to_id)
    dev_data = process_sst2_data("../Section7/SST-2/dev.tsv", word_to_id)
    X_train, y_train = create_mean_embedding_features(train_data, embedding_matrix)
    X_dev, y_dev = create_mean_embedding_features(dev_data, embedding_matrix)
    model = MeanEmbeddingClassifier(embedding_matrix.size(1))

    print("モデル構造:")
    print(model)
    print(f"\n特徴ベクトルの形状: {X_train.shape}")
    print(f"正解ラベルの形状: {y_train.shape}")


if __name__ == "__main__":
    main()