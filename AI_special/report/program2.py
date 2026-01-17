import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import shap

# カラム名リスト
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]

# データ読み込み
file_path = 'agaricus-lepiota.data'
df = pd.read_csv(file_path, header=None, names=columns)

# 欠損値（?）をNaNに置換
df = df.replace('?', pd.NA)

# 欠損値のある行を削除（または他の処理も可）
df = df.dropna()

# 目的変数と説明変数に分割
y = df['class'].map({'e': 0, 'p': 1})  # e:食用=0, p:毒=1
X = df.drop('class', axis=1)

# カテゴリ変数をダミー変数化
X = pd.get_dummies(X)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストで学習
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 予測と評価
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 代表的な特徴量を選択（例: odor）
# ダミー変数化後のカラム名を取得
odor_cols = [col for col in X.columns if col.startswith('odor_')]

# PDP, ICEの描画
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    clf, X_test, features=odor_cols, kind="both", ax=ax
)
plt.suptitle('PDP & ICE for odor')
plt.tight_layout()

# SHAP値の計算
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot（全特徴量の重要度）
shap.summary_plot(shap_values[1], X_test, plot_type="bar")
shap.summary_plot(shap_values[1], X_test)
