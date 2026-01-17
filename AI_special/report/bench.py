import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def create_features(data, target_col, benchmark_col):
    """改善された特徴量作成関数"""
    df = data.copy()
    
    # 基本リターン（Look-ahead biasを修正）
    df['target_return_1d'] = df[target_col].pct_change()
    df['benchmark_return_1d'] = df[benchmark_col].pct_change()
    
    # 複数期間のリターン
    for period in [3, 5, 10, 20]:
        df[f'target_return_{period}d'] = df[target_col].pct_change(period)
        df[f'benchmark_return_{period}d'] = df[benchmark_col].pct_change(period)
    
    # ボラティリティ指標
    for period in [5, 10, 20]:
        df[f'target_vol_{period}d'] = df['target_return_1d'].rolling(period).std()
        df[f'benchmark_vol_{period}d'] = df['benchmark_return_1d'].rolling(period).std()
    
    # 移動平均との乖離
    for period in [5, 10, 20]:
        df[f'target_ma_diff_{period}d'] = (df[target_col] - df[target_col].rolling(period).mean()) / df[target_col].rolling(period).mean()
        df[f'benchmark_ma_diff_{period}d'] = (df[benchmark_col] - df[benchmark_col].rolling(period).mean()) / df[benchmark_col].rolling(period).mean()
    
    # 相対パフォーマンス
    df['relative_return_1d'] = df['target_return_1d'] - df['benchmark_return_1d']
    df['relative_return_5d'] = df['target_return_5d'] - df['benchmark_return_5d']
    
    # RSI（Relative Strength Index）
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['target_rsi'] = calculate_rsi(df[target_col])
    df['benchmark_rsi'] = calculate_rsi(df[benchmark_col])
    
    # 出来高関連（利用可能な場合）
    if 'Volume' in data.columns:
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df

def evaluate_model_with_time_series_cv(X, y, model, n_splits=5):
    """時系列交差検証での評価"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for train_idx, val_idx in tscv.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # SMOTE適用（クラス不均衡対策）
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_fold)
        
        # モデル訓練・予測
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_val_scaled)
        
        # スコア計算
        scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
        scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
        scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
        scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
    
    return {metric: np.mean(values) for metric, values in scores.items()}

# メイン処理
if __name__ == "__main__":
    # データ取得
    target_symbol = "2432.T"  # DeNA
    benchmark_symbol = "1306.T"  # TOPIX連動型上場投資信託
    
    try:
        # データダウンロード
        symbols = [target_symbol, benchmark_symbol]
        data_raw = yf.download(symbols, start="2020-07-25", end="2025-07-25", progress=False)['Close']
        
        # データ結合
        data = pd.DataFrame({
            'target': data_raw[target_symbol],
            'benchmark': data_raw[benchmark_symbol]
        }).dropna()
        
        print(f"使用銘柄: {target_symbol} vs {benchmark_symbol}")
        print(f"データ期間: {data.index[0]} 〜 {data.index[-1]}")
        print(f"総データ数: {len(data)}日")
        
        # 特徴量作成
        featured_data = create_features(data, 'target', 'benchmark')
        
        # ターゲット作成（翌日のアウトパフォーマンス予測）
        featured_data['target_return_next'] = featured_data['target'].pct_change().shift(-1)
        featured_data['benchmark_return_next'] = featured_data['benchmark'].pct_change().shift(-1)
        featured_data['label'] = (featured_data['target_return_next'] > featured_data['benchmark_return_next']).astype(int)
        
        # 欠損値除去
        featured_data = featured_data.dropna()
        
        # 特徴量選択
        feature_columns = [col for col in featured_data.columns if col not in ['target', 'benchmark', 'target_return_next', 'benchmark_return_next', 'label']]
        
        X = featured_data[feature_columns]
        y = featured_data['label']
        
        print(f"特徴量数: {len(feature_columns)}")
        print(f"クラス分布:")
        print(y.value_counts(normalize=True))
        
        # モデル定義
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(random_state=42),
            "KNN": KNeighborsClassifier(),
            "MLP": MLPClassifier(max_iter=1000, random_state=42)
        }
        
        # 時系列交差検証での評価
        print("\n=== 時系列交差検証結果 ===")
        results = {}
        for name, model in models.items():
            scores = evaluate_model_with_time_series_cv(X, y, model)
            results[name] = scores
            print(f"\n{name}:")
            for metric, score in scores.items():
                print(f"  {metric}: {score:.3f}")
        
        # 結果の可視化用DataFrame
        results_df = pd.DataFrame(results).T
        print(f"\n=== 全モデル比較 ===")
        print(results_df.round(3))
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("1306.T（TOPIX ETF）をベンチマークとして使用しています")