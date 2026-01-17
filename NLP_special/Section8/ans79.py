import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Set
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sst2_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep="\t", header=0)


def get_vocabulary(df: pd.DataFrame) -> Set[str]:
    vocabulary = set()
    for text in df["sentence"]:
        vocabulary.update(text.lower().split())
    return vocabulary


class TextClassifier(nn.Module):
    def __init__(
        self, embedding_matrix: torch.Tensor, hidden_dim: int = 256, num_layers: int = 2
    ):
        super().__init__()
        embedding_dim = embedding_matrix.size(1)

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.embedding(x)

        x = x.transpose(1, 2)  
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.fc(x)

        return x


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

    return {"input_ids": input_tensor.to(device), "label": label_tensor.to(device)}


class SST2Dataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

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
    num_epochs: int = 10,
    learning_rate: float = 0.001,
) -> None:
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # 訓練モード
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(batch["input_ids"])
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
                outputs = model(batch["input_ids"])
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
    torch.save(model.state_dict(), "Section8/model.pth")


def evaluate_model(model: nn.Module, dev_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dev_loader:
            outputs = model(batch["input_ids"])
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

    train_dataset = SST2Dataset(train_data)
    dev_dataset = SST2Dataset(dev_data)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=32, shuffle=False, collate_fn=collate
    )
    model = TextClassifier(embedding_matrix).to(device)
    train_model(model, train_loader, dev_loader)
    accuracy = evaluate_model(model, dev_loader)
    print(f"\n開発セットの正解率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()