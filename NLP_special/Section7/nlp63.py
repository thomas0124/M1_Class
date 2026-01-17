import pandas as pd
import pickle
from collections import defaultdict

def add_feature(sentence, label):
    data = {"sentence": sentence, "label": label, "feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data

with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

df_dev = pd.read_csv("SST-2/dev.tsv", sep="\t")

first_sentence = df_dev["sentence"].iloc[0]
first_label = df_dev["label"].iloc[0]

data = add_feature(first_sentence, first_label)

X = vec.transform([data["feature"]])

import pandas as pd
df = pd.DataFrame(X, columns=vec.get_feature_names_out())
print(df["charming"])

predicted_label = model.predict(X)[0]
predicted_prob = model.predict_proba(X)[0]

print(f"文: {first_sentence}")
print(f"実際のラベル: {first_label}")
print(f"予測されたラベル: {predicted_label}")
print(f"予測は正しいか: {predicted_label == first_label}")
