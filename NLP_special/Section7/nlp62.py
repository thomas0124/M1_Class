import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
import pickle

def add_feature(sentence, label):
    data = {"sentence": sentence, "label": label, "feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data

df_train = pd.read_csv("SST-2/train.tsv", sep="\t")
df_dev = pd.read_csv("SST-2/dev.tsv", sep="\t")

data_train = []
for sentence, label in zip(df_train["sentence"], df_train["label"]):
    data_train.append(add_feature(sentence, label))

data_dev = []
for sentence, label in zip(df_dev["sentence"], df_dev["label"]):
    data_dev.append(add_feature(sentence, label))

vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform([d["feature"] for d in data_train])
y_train = [d["label"] for d in data_train]
X_dev = vec.transform([d["feature"] for d in data_dev])
y_dev = [d["label"] for d in data_dev]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vec, f)