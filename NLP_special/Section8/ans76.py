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


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)

    batch_size = len(batch)
    input_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
    label_tensor = torch.zeros((batch_size, 1), dtype=torch.float)

    lengths = [len(item["input_ids"]) for item in batch]

    sorted_indices = sorted(range(batch_size), key=lambda i: lengths[i], reverse=True)

    for i, idx in enumerate(sorted_indices):
        item = batch[idx]
        input_tensor[i, : len(item["input_ids"])] = item["input_ids"]
        label_tensor[i] = item["label"]

    return {"input_ids": input_tensor, "label": label_tensor}


class SST2Dataset(Dataset):
    def __init__(self, data: List[Dict], embedding_matrix: torch.Tensor):
        self.data = data
        self.embedding_matrix = embedding_matrix

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


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
    embedding_matrix: torch.Tensor,
    num_epochs: int = 10,
    learning_rate: float = 0.01,
) -> None:
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            embeddings = embedding_matrix[
                batch["input_ids"]
            ]
            mean_embeddings = torch.mean(
                embeddings, dim=1
            )

            outputs = model(mean_embeddings)
            loss = criterion(outputs, batch["label"])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch["label"].size(0)
            train_correct += (predicted == batch["label"]).sum().item()
        model.eval()
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0

        with torch.no_grad():
            for batch in dev_loader:
                embeddings = embedding_matrix[batch["input_ids"]]
                mean_embeddings = torch.mean(embeddings, dim=1)

                outputs = model(mean_embeddings)
                loss = criterion(outputs, batch["label"])

                dev_loss += loss.item()
                predicted = (outputs > 0.5).float()
                dev_total += batch["label"].size(0)
                dev_correct += (predicted == batch["label"]).sum().item()
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


def evaluate_model(
    model: nn.Module, dev_loader: DataLoader, embedding_matrix: torch.Tensor
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dev_loader:
            embeddings = embedding_matrix[batch["input_ids"]]
            mean_embeddings = torch.mean(embeddings, dim=1)

            outputs = model(mean_embeddings)
            predicted = (outputs > 0.5).float()
            total += batch["label"].size(0)
            correct += (predicted == batch["label"]).sum().item()

    accuracy = 100 * correct / total
    return accuracy


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
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=8, shuffle=False, collate_fn=collate
    )
    model = MeanEmbeddingClassifier(embedding_matrix.size(1))
    train_model(model, train_loader, dev_loader, embedding_matrix)
    accuracy = evaluate_model(model, dev_loader, embedding_matrix)
    print(f"\n開発セットの正解率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()