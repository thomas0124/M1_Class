import pickle

with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

feature_names = vec.get_feature_names_out()

weights = model.coef_[0]

print(weights)

weight_feature_pairs = list(zip(weights, feature_names))
print(weight_feature_pairs)

top_20_positive = sorted(weight_feature_pairs, key=lambda x: x[0], reverse=True)[:20]

top_20_negative = sorted(weight_feature_pairs, key=lambda x: x[0])[:20]

print("重みの高い特徴量トップ20:")
for i, (weight, feature) in enumerate(top_20_positive, 1):
    print(f"{i}. {feature}: {weight:.4f}")

print("\n重みの低い特徴量トップ20:")
for i, (weight, feature) in enumerate(top_20_negative, 1):
    print(f"{i}. {feature}: {weight:.4f}")


