import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

target = "7203.T"
topix = "1306.T"

start = "2020-01-01"
end = "2025-12-31"

data = yf.download([target, topix], start=start, end=end, interval="1d", auto_adjust=True)['Close']

data = data.dropna()

returns = data.pct_change().shift(-1)

returns.columns = ['target_return', 'topix_return']

returns['label'] = (returns['target_return'] > returns['topix_return']).astype(int)

returns['target_return_today'] = data[target].pct_change()
returns['topix_return_today'] = data[topix].pct_change()
returns['target_return_5d'] = data[target].pct_change(5)
returns['topix_return_5d'] = data[topix].pct_change(5)
returns['target_return_3d'] = data[target].pct_change(3)
returns['topix_return_3d'] = data[topix].pct_change(3)
returns['target_vol_5d'] = data[target].pct_change().rolling(5).std()
returns['topix_vol_5d'] = data[topix].pct_change().rolling(5).std()
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

test = returns[(returns.index >= "2025-01-01") & (returns.index < "2026-01-01")]

X_train = train[features]
y_train = train['label']
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

for name, model in models.items():
    if name == "XGBoost":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"{name} F1 Score: {f1_score(y_test, y_pred):.3f}")