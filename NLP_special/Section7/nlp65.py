import pickle
from collections import defaultdict


def add_feature(sentence):
    data = {"feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data

def predict_sentiment(text):
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)

    data = add_feature(text)

    X = vec.transform([data["feature"]])

    predicted_label = model.predict(X)[0]
    predicted_prob = model.predict_proba(X)[0]

    sentiment = "ポジティブ" if predicted_label == 1 else "ネガティブ"
    print(f"テキスト: {text}")
    print(f"予測された感情: {sentiment}")
    print(
        f"予測確率: ネガティブ={predicted_prob[0]:.4f}, ポジティブ={predicted_prob[1]:.4f}"
    )

test_text = "the worst movie I 've ever seen"
predict_sentiment(test_text)

def interactive_prediction():
    print(
        "\n対話的にテキストを入力して予測します。終了するには 'q' を入力してください。"
    )
    while True:
        text = input("\nテキストを入力してください: ")
        if text.lower() == "q":
            break
        predict_sentiment(text)


if __name__ == "__main__":
    interactive_prediction()