import numpy as np
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# データセットのロード（RDKitDescriptorsで特徴量化）
featurizer = dc.feat.RDKitDescriptors()
tasks, datasets, transformers = dc.molnet.load_bace_regression(featurizer)
train_set, val_set, test_set = datasets

# データの準備
X_train, y_train = train_set.X, train_set.y.ravel()
X_val, y_val = val_set.X, val_set.y.ravel()
X_test, y_test = test_set.X, test_set.y.ravel()

# ランダムフォレスト回帰モデルの学習
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 評価指標
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
print(f"Test R^2: {r2:.4f}")

# 散布図で予測値と実測値の比較
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True normalized pIC50')
plt.ylabel('Predicted normalized pIC50')
plt.title(f'RandomForest Test MSE={mse:.3f}, R2={r2:.3f}')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()

# ---
# 考察例：
# ランダムフォレスト回帰は非線形性を捉えやすく、特徴量のスケーリングや前処理に強い。
# ニューラルネットワークよりも少ないデータでも安定した予測が可能な場合が多い。
# 今回のデータセットでは、順伝播型NNよりも予測精度が向上することが期待できる。
