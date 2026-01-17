import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定（matplotlib用）
plt.rcParams['font.family'] = 'DejaVu Sans'

print("="*80)
print("東証プライム企業株価予測モデル構築レポート")
print("="*80)

# 1. 企業選択と基本情報
target_code = "2432.T"  # ディー・エヌ・エー（DeNA）
topix_code = "1306.T"   # TOPIX連動型上場投資信託
symbols = (topix_code, target_code)

print(f"\n【1. 分析対象企業】")
print(f"企業コード: {target_code}")
print(f"企業名: 株式会社ディー・エヌ・エー（DeNA）")
print(f"ベンチマーク: {topix_code} (TOPIX連動型ETF)")

# 2. データ取得
print(f"\n【2. データ取得】")
try:
    data = yf.download(symbols, period='5y', interval="1d")['Close']
    data = data.dropna()
    print(f"取得期間: {data.index[0].strftime('%Y-%m-%d')} ～ {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"総データ数: {len(data)} 日")
except Exception as e:
    print(f"データ取得エラー: {e}")
    exit(1)

# 3. 特徴量エンジニアリング
print(f"\n【3. 特徴量エンジニアリング】")

# リターン計算（翌日のリターン）
returns = data.pct_change().shift(-1)
returns = returns.dropna()
returns.columns = ['topix_return_next', 'target_return_next']

# ラベル作成（翌日のターゲット企業リターンがTOPIXを上回るかどうか）
returns['label'] = (returns['target_return_next'] > returns['topix_return_next']).astype(int)

# 特徴量作成
# 1. 当日リターン
returns['target_return_today'] = data[target_code].pct_change()
returns['topix_return_today'] = data[topix_code].pct_change()

# 2. 過去n日リターン
for period in [3, 5, 10]:
    returns[f'target_return_{period}d'] = data[target_code].pct_change(period)
    returns[f'topix_return_{period}d'] = data[topix_code].pct_change(period)

# 3. ボラティリティ（過去5日、10日の標準偏差）
for period in [5, 10]:
    returns[f'target_vol_{period}d'] = data[target_code].pct_change().rolling(period).std()
    returns[f'topix_vol_{period}d'] = data[topix_code].pct_change().rolling(period).std()

# 4. 移動平均からの乖離率
for period in [5, 10, 20]:
    returns[f'ma_diff_{period}d'] = (data[target_code] - data[target_code].rolling(period).mean()) / data[target_code].rolling(period).mean()

# 5. リターン差分
returns['ret_diff_today'] = returns['target_return_today'] - returns['topix_return_today']
returns['ret_diff_3d'] = returns['target_return_3d'] - returns['topix_return_3d']
returns['ret_diff_5d'] = returns['target_return_5d'] - returns['topix_return_5d']

# 6. 相対強度指標（RSI風）
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

returns['target_rsi'] = calculate_rsi(data[target_code])
returns['topix_rsi'] = calculate_rsi(data[topix_code])

# NaN値除去
returns = returns.dropna()

# 特徴量リスト
features = [
    'target_return_today', 'topix_return_today',
    'target_return_3d', 'topix_return_3d',
    'target_return_5d', 'topix_return_5d',
    'target_return_10d', 'topix_return_10d',
    'target_vol_5d', 'topix_vol_5d',
    'target_vol_10d', 'topix_vol_10d',
    'ma_diff_5d', 'ma_diff_10d', 'ma_diff_20d',
    'ret_diff_today', 'ret_diff_3d', 'ret_diff_5d',
    'target_rsi', 'topix_rsi'
]

print(f"作成した特徴量数: {len(features)}")
print("特徴量リスト:")
for i, feature in enumerate(features, 1):
    print(f"  {i:2d}. {feature}")

# 4. データ分割
print(f"\n【4. データ分割】")
train_data = returns[returns.index < "2025-01-01"]
test_data = returns[returns.index >= "2025-01-01"]

X_train = train_data[features]
y_train = train_data['label']
X_test = test_data[features]
y_test = test_data['label']

print(f"訓練データ期間: {train_data.index[0].strftime('%Y-%m-%d')} ～ {train_data.index[-1].strftime('%Y-%m-%d')}")
print(f"訓練データ数: {len(X_train)} 日")
print(f"検証データ期間: {test_data.index[0].strftime('%Y-%m-%d')} ～ {test_data.index[-1].strftime('%Y-%m-%d')}")
print(f"検証データ数: {len(X_test)} 日")

# ラベル分布確認
train_positive_rate = y_train.mean()
test_positive_rate = y_test.mean()
print(f"\n訓練データにおけるアウトパフォーム率: {train_positive_rate:.3f}")
print(f"検証データにおけるアウトパフォーム率: {test_positive_rate:.3f}")

# 5. 前処理（標準化）
print(f"\n【5. データ前処理】")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("特徴量の標準化を実行しました")

# 6. モデル構築と評価
print(f"\n【6. モデル構築と評価】")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # データの選択（SVMとMLPは標準化版を使用）
    if name in ["SVM", "MLP", "LogisticRegression"]:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # モデル訓練
    model.fit(X_train_use, y_train)
    
    # 予測
    y_pred = model.predict(X_test_use)
    y_pred_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 評価指標計算
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")

# 7. 結果比較
print(f"\n【7. モデル性能比較】")
print("="*70)
print(f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
print("="*70)

for name, metrics in results.items():
    print(f"{name:<20} {metrics['accuracy']:<10.3f} {metrics['f1_score']:<10.3f} "
          f"{metrics['precision']:<10.3f} {metrics['recall']:<10.3f}")

# 最良モデルの特定
best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
print(f"\n最高F1スコア: {best_model_name} ({results[best_model_name]['f1_score']:.3f})")

# 8. 詳細分析（最良モデル）
print(f"\n【8. 最良モデル詳細分析: {best_model_name}】")

best_predictions = results[best_model_name]['y_pred']

# 混同行列
print("\n混同行列:")
cm = confusion_matrix(y_test, best_predictions)
print(f"           予測")
print(f"         0    1")
print(f"実際 0  {cm[0,0]:3d}  {cm[0,1]:3d}")
print(f"     1  {cm[1,0]:3d}  {cm[1,1]:3d}")

# 分類レポート
print(f"\n分類レポート:")
print(classification_report(y_test, best_predictions, target_names=['アンダーパフォーム', 'アウトパフォーム']))

# 9. 特徴量重要度分析（RandomForestの場合）
if best_model_name == "RandomForest":
    print(f"\n【9. 特徴量重要度分析】")
    feature_importance = models[best_model_name].feature_importances_
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("上位10の重要特徴量:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<20} {row['importance']:.4f}")

# 10. 実際のリターン分析
print(f"\n【10. 実際のリターン分析】")
test_returns = test_data[['target_return_next', 'topix_return_next', 'label']].copy()
test_returns['prediction'] = best_predictions

# 予測が当たった場合の分析
correct_predictions = test_returns[test_returns['label'] == test_returns['prediction']]
accuracy_rate = len(correct_predictions) / len(test_returns)

print(f"予測精度: {accuracy_rate:.3f}")
print(f"予測が当たった日数: {len(correct_predictions)} / {len(test_returns)}")

# アウトパフォーム予測が当たった場合の平均リターン差
outperform_correct = test_returns[(test_returns['label'] == 1) & (test_returns['prediction'] == 1)]
if len(outperform_correct) > 0:
    avg_return_diff = (outperform_correct['target_return_next'] - outperform_correct['topix_return_next']).mean()
    print(f"アウトパフォーム予測が的中した場合の平均リターン差: {avg_return_diff:.4f} ({avg_return_diff*100:.2f}%)")

# 11. 考察
print(f"\n【11. 考察】")
print("="*50)

print("\n(1) モデル性能について:")
print(f"   - 最も優秀だったモデル: {best_model_name}")
print(f"   - F1スコア: {results[best_model_name]['f1_score']:.3f}")
print(f"   - 精度: {results[best_model_name]['accuracy']:.3f}")

if results[best_model_name]['accuracy'] > 0.5:
    print("   - ランダム予測（50%）を上回る性能を示している")
else:
    print("   - ランダム予測と同程度かそれ以下の性能")

print(f"\n(2) データ特性について:")
print(f"   - 訓練データでのアウトパフォーム率: {train_positive_rate:.3f}")
print(f"   - 検証データでのアウトパフォーム率: {test_positive_rate:.3f}")

if abs(train_positive_rate - test_positive_rate) > 0.1:
    print("   - 訓練・検証期間でラベル分布に大きな差がある")
else:
    print("   - 訓練・検証期間でラベル分布は比較的安定")

print(f"\n(3) 予測の困難さ:")
print("   - 株価予測は本質的に困難なタスク")
print("   - 市場の効率性により、過去の情報から将来を予測することは限界がある")
print("   - ランダムウォーク仮説により、短期的な価格変動は予測困難")

print(f"\n(4) 改善提案:")
print("   - より多様な特徴量の追加（マクロ経済指標、センチメント指標等）")
print("   - より長期間のデータの活用")
print("   - アンサンブル手法の適用")
print("   - 異なる期間での交差検証の実施")
print("   - リスク調整済みリターンでの評価")

print("\n" + "="*80)
print("レポート完了")
print("="*80)