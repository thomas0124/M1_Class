import yfinance as yf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

target = "2270.T"
topix = ("1306.T", target)

data = yf.download(topix, period="5y", interval="1d")['Close']
data = data.dropna()

returns = data.pct_change().shift(-1)
returns = data.dropna()

returns.columns = ['TOPIX', 'Target']
returns['label'] = (returns['Target'] > returns['TOPIX']).astype(int)

returns["target_2d"] = data[target].pct_change()
returns["topix_2d"] = data[topix[0]].pct_change()
returns["target_3d"] = data[target].pct_change(3)
returns["topix_3d"] = data[topix[0]].pct_change(3)
returns["target_4d"] = data[target].pct_change(4)
returns["topix_4d"] = data[topix[0]].pct_change(4)
returns["target_5d"] = data[target].pct_change(5)
returns["topix_5d"] = data[topix[0]].pct_change(5)
returns["sub"] = returns["target_2d"] - returns["topix_2d"]

returns = returns.dropna()

feautures = [
    "target_2d","topix_2d",
    "target_3d","topix_3d",
    "target_4d","topix_4d",
    "target_5d","topix_5d",
    "sub"
]

train = returns[returns.index < "2025-01-01"]
test = returns[returns.index >= "2025-01-01"]

X_train = train[feautures]
y_train = train['label']
X_test = test[feautures]
y_test = test['label']

print(y_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
} 

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("\n")