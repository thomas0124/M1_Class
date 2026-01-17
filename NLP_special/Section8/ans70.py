import numpy as np
from gensim.models import KeyedVectors
from typing import Dict, Tuple


def load_word_embeddings(
    model_path: str = "../Section6/GoogleNews-vectors-negative300.bin",
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    vocab_size = len(model.key_to_index)
    embedding_dim = model.vector_size

    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))

    word_to_id = {"<PAD>": 0}
    id_to_word = {0: "<PAD>"}

    for i, word in enumerate(model.key_to_index, start=1):
        embedding_matrix[i] = model[word]
        word_to_id[word] = i
        id_to_word[i] = word

    return embedding_matrix, word_to_id, id_to_word


def main():
    embedding_matrix, word_to_id, id_to_word = load_word_embeddings()
    print(f"単語埋め込み行列の形状: {embedding_matrix.shape}")
    print(f"語彙数: {len(word_to_id)}")
    print(f"埋め込み次元数: {embedding_matrix.shape[1]}")
    print("\n最初の5単語:")
    for i in range(1, 6):
        word = id_to_word[i]
        print(f"ID: {i}, 単語: {word}, ベクトル: {embedding_matrix[i][:5]}...")


if __name__ == "__main__":
    main()