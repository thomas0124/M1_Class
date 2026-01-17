import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score,  recall_score, precision_score, confusion_matrix

target = "2432.T"
symbols = ("1306.T", target)

data = yf.download(symbols, period='5y', interval = "1d")['Close']
data = data.dropna()

returns = data.pct_change().shift(-1)
returns = returns.dropna()

returns.columns = ['target_return', 'topix_return']

returns['label'] = (returns['target_return'] > returns['topix_return']).astype(int)

returns['target_return_today'] = data[target].pct_change()
returns['topix_return_today'] = data[symbols[0]].pct_change()
returns['target_return_5d'] = data[target].pct_change(5)
returns['topix_return_5d'] = data[symbols[0]].pct_change(5)
returns['target_return_3d'] = data[target].pct_change(3)
returns['topix_return_3d'] = data[symbols[0]].pct_change(3)
returns['target_vol_5d'] = data[target].pct_change().rolling(5).std()
returns['topix_vol_5d'] = data[symbols[0]].pct_change().rolling(5).std()
returns['ma_diff_5d'] = (data[target] - data[target].rolling(5).mean()) / data[target].rolling(5).mean()
returns['ret_diff'] = returns['target_return_today'] - returns['topix_return_today']
returns = returns.dropna()

features = [
    'target_return_today', 'topix_return_today',
    'target_return_3d', 'topix_return_3d',
    'target_return_5d', 'topix_return_5d',
    'target_vol_5d', 'topix_vol_5d',
    'ma_diff_5d', 'ret_diff'
]

train = returns[returns.index < "2025-01-01"]
test = returns[returns.index >= "2025-01-01"]

X_train = train[features]

y_train = train['label']

print()

# Positiveサンプル数のカウント
p_count = 0
n_count = 0
for i in range(len(train)):
    flag = train['label'].iloc[i]
    if flag == 1:
        p_count += 1
    else:
        n_count += 1
print(f"Positive samples in training set: {p_count}")
print(f"Negative samples in training set: {n_count}")


X_test = test[features]
y_test = test['label']

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

max_scores = {
    "Accuracy": (0, ""),
    "Precision": (0, ""),
    "Recall": (0, ""),
    "F1": (0, "")
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"{name} Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"{name} Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"{name} F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print("-" * 40)
    if acc > max_scores["Accuracy"][0]:
        max_scores["Accuracy"] = (acc, name)
    if prec > max_scores["Precision"][0]:
        max_scores["Precision"] = (prec, name)
    if rec > max_scores["Recall"][0]:
        max_scores["Recall"] = (rec, name)
    if f1 > max_scores["F1"][0]:
        max_scores["F1"] = (f1, name)
        
print("=== 各評価指標の最大値 ===")
for metric, (score, model_name) in max_scores.items():
    print(f"{metric}: {score:.3f} ({model_name})")