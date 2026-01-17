import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

embeddings = []
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    mean_embedding = torch.mean(outputs.last_hidden_state[0], dim=0).numpy()
    embeddings.append(mean_embedding)

embeddings = np.array(embeddings)

similarity_matrix = cosine_similarity(embeddings)

print("文の組み合わせに対するコサイン類似度:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(
            f"'{sentences[i]}' と '{sentences[j]}' の類似度: {similarity_matrix[i][j]:.4f}"
        )