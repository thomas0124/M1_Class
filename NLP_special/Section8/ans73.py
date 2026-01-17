import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Set
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset


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


class SST2Dataset(Dataset):
    def __init__(self, data: List[Dict], embedding_matrix: torch.Tensor):
        self.data = data
        self.embedding_matrix = embedding_matrix

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        input_ids = item["input_ids"]
        embeddings = self.embedding_matrix[input_ids]
        mean_embedding = torch.mean(embeddings, dim=0)
        return mean_embedding, item["label"]


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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.01,
) -> None:
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # 訓練モード
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        model.eval()
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0

        with torch.no_grad():
            for inputs, labels in dev_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                dev_loss += loss.item()
                predicted = (outputs > 0.5).float()
                dev_total += labels.size(0)
                dev_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * train_correct / train_total
        dev_accuracy = 100 * dev_correct / dev_total
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Train Acc: {train_accuracy:.2f}%, "
            f"Dev Loss: {dev_loss / len(dev_loader):.4f}, "
            f"Dev Acc: {dev_accuracy:.2f}%"
        )
    torch.save(model.state_dict(), "./model.pth")


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
    train_dataset = SST2Dataset(train_data, embedding_matrix)
    dev_dataset = SST2Dataset(dev_data, embedding_matrix)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    model = MeanEmbeddingClassifier(embedding_matrix.size(1))

    # モデルの学習
    train_model(model, train_loader, dev_loader)


if __name__ == "__main__":
    main()