import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
import shap
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

# カラム名リスト
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]

def load_and_preprocess_data():
    """データの読み込みと前処理"""
    print("=== データ読み込みと前処理 ===")
    
    # データ読み込み
    file_path = 'agaricus-lepiota.data'
    df = pd.read_csv(file_path, header=None, names=columns)
    
    print(f"データ形状: {df.shape}")
    print(f"クラス分布:\n{df['class'].value_counts()}")
    
    # 欠損値（?）をNaNに置換
    df = df.replace('?', pd.NA)
    
    # 欠損値の確認
    missing_values = df.isnull().sum()
    print(f"\n欠損値:\n{missing_values[missing_values > 0]}")
    
    # 欠損値のある行を削除
    df_clean = df.dropna()
    print(f"欠損値削除後のデータ形状: {df_clean.shape}")
    
    return df_clean

def encode_categorical_features(df):
    """カテゴリ変数のエンコーディング"""
    print("\n=== カテゴリ変数のエンコーディング ===")
    
    # 目的変数のエンコーディング
    y = (df['class'] == 'p').astype(int)  # p: 毒=1, e: 食用=0
    
    # 説明変数から目的変数を除外
    X = df.drop('class', axis=1)
    
    # カテゴリ変数をダミー変数化
    X_encoded = pd.get_dummies(X, drop_first=False)
    
    print(f"エンコーディング後の特徴量数: {X_encoded.shape[1]}")
    
    return X_encoded, y

def train_model(X, y):
    """モデルの訓練"""
    print("\n=== モデル訓練 ===")
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ランダムフォレストで学習
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1
    )
    
    # クロスバリデーション
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"クロスバリデーション精度: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # モデル訓練
    rf_model.fit(X_train, y_train)
    
    # 予測と評価
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    print(f"テストセット精度: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred, target_names=['食用', '毒']))
    
    return rf_model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba

def plot_feature_importance(model, X, top_n=15):
    """特徴量重要度の可視化"""
    print("\n=== 特徴量重要度 ===")
    
    # 特徴量重要度を取得
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 上位N個の特徴量をプロット
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('重要度')
    plt.title(f'ランダムフォレスト - 上位{top_n}特徴量の重要度')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"上位{top_n}特徴量:")
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance

def plot_confusion_matrix(y_test, y_pred):
    """混同行列の可視化"""
    print("\n=== 混同行列 ===")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['食用', '毒'], yticklabels=['食用', '毒'])
    plt.title('混同行列')
    plt.ylabel('実際のクラス')
    plt.xlabel('予測クラス')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 詳細な分析
    tn, fp, fn, tp = cm.ravel()
    print(f"真陰性 (TN): {tn} - 食用を正しく食用と予測")
    print(f"偽陽性 (FP): {fp} - 食用を毒と誤予測")
    print(f"偽陰性 (FN): {fn} - 毒を食用と誤予測")
    print(f"真陽性 (TP): {tp} - 毒を正しく毒と予測")
    print(f"精度: {(tp + tn) / (tp + tn + fp + fn):.4f}")
    print(f"再現率: {tp / (tp + fn):.4f}")
    print(f"特異度: {tn / (tn + fp):.4f}")

def plot_pdp_ice(model, X_test, feature_names, top_features, n_features=5):
    """PDPとICEの可視化"""
    print("\n=== PDP (Partial Dependence Plot) と ICE (Individual Conditional Expectation) ===")
    
    # 上位特徴量のPDPとICEをプロット
    top_feature_names = [f for f in top_features['feature'].head(n_features)]
    
    # 利用可能な特徴量を確認
    available_features = [f for f in top_feature_names if f in X_test.columns]
    print(f"利用可能な特徴量: {available_features}")
    
    if len(available_features) == 0:
        print("PDP/ICE描画に適した特徴量が見つかりませんでした。")
        return
    
    # サブプロット数を調整
    n_plots = min(len(available_features), 6)
    n_rows = (n_plots + 2) // 3  # 3列で配置
    n_cols = min(n_plots, 3)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(available_features[:n_plots]):
        try:
            # 特徴量の値の分布を確認
            feature_values = X_test[feature].unique()
            print(f"特徴量 {feature} の値: {feature_values}")
            
            # 手動でPDPを計算
            feature_0_pred = model.predict_proba(X_test[X_test[feature] == 0])[:, 1].mean()
            feature_1_pred = model.predict_proba(X_test[X_test[feature] == 1])[:, 1].mean()
            
            # プロット
            axes[i].bar([0, 1], [feature_0_pred, feature_1_pred], color=['blue', 'red'])
            axes[i].set_title(f'PDP: {feature}')
            axes[i].set_xlabel('特徴量の値')
            axes[i].set_ylabel('平均予測確率（毒）')
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(['False', 'True'])
            
        except Exception as e:
            print(f"特徴量 {feature} のPDP描画でエラー: {e}")
            axes[i].text(0.5, 0.5, f'Error: {feature}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'PDP: {feature} (Error)')
    
    # 余分なサブプロットを削除
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('pdp_ice_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def apply_shap_analysis(model, X_test, feature_names, y_test, y_pred_proba):
    """SHAP分析の適用"""
    print("\n=== SHAP分析 ===")
    
    try:
        # SHAP値の計算
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # SHAP summary plot（全特徴量の重要度）
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
        plt.title('SHAP - 特徴量重要度')
        plt.tight_layout()
        plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # SHAP summary plot（詳細）
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[1], X_test, show=False)
        plt.title('SHAP - 特徴量の影響')
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 個別サンプルのSHAP値
        print("個別サンプルのSHAP分析（最初の5サンプル）:")
        for i in range(min(5, len(X_test))):
            print(f"\nサンプル {i+1}:")
            print(f"実際のクラス: {'毒' if y_test.iloc[i] == 1 else '食用'}")
            print(f"予測確率: {y_pred_proba[i]:.4f}")
            
            # 上位10個のSHAP値
            sample_shap = shap_values[1][i]
            feature_shap = list(zip(X_test.columns, sample_shap))
            feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("上位10個の特徴量のSHAP値:")
            for feature, shap_val in feature_shap[:10]:
                print(f"  {feature}: {shap_val:.4f}")
                
    except Exception as e:
        print(f"SHAP分析でエラーが発生しました: {e}")
        print("SHAP分析をスキップします。")

def analyze_specific_features(model, X_test, y_test, y_pred_proba):
    """特定の特徴量の詳細分析"""
    print("\n=== 特定特徴量の詳細分析 ===")
    
    # 重要な特徴量の分析
    important_features = ['odor_n', 'odor_f', 'spore-print-color_r', 'gill-size_n']
    
    for feature in important_features:
        if feature in X_test.columns:
            print(f"\n特徴量: {feature}")
            
            # この特徴量が1のサンプルを抽出
            feature_samples = X_test[X_test[feature] == 1]
            if len(feature_samples) > 0:
                # 安全なインデックス取得
                feature_indices = feature_samples.index
                
                # y_testとy_pred_probaの対応する値を取得
                feature_y_true = y_test.loc[feature_indices]
                feature_y_pred = y_pred_proba[feature_indices.values]
                
                print(f"  この特徴量を持つサンプル数: {len(feature_samples)}")
                print(f"  実際の毒の割合: {feature_y_true.mean():.4f}")
                print(f"  平均予測確率: {feature_y_pred.mean():.4f}")
                
                # 高確率で毒と予測されたサンプル
                high_poison_pred = feature_y_pred > 0.8
                if high_poison_pred.sum() > 0:
                    print(f"  高確率で毒と予測されたサンプル: {high_poison_pred.sum()}")
                    print(f"  その中で実際に毒だった割合: {feature_y_true.iloc[high_poison_pred].mean():.4f}")

def main():
    """メイン関数"""
    print("キノコ毒性予測モデルの構築と解釈分析")
    print("=" * 50)
    
    # データ読み込みと前処理
    df = load_and_preprocess_data()
    
    # カテゴリ変数のエンコーディング
    X, y = encode_categorical_features(df)
    
    # モデル訓練
    model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_model(X, y)
    
    # 特徴量重要度の可視化
    feature_importance = plot_feature_importance(model, X)
    
    # 混同行列の可視化
    plot_confusion_matrix(y_test, y_pred)
    
    # PDPとICEの可視化
    plot_pdp_ice(model, X_test, X.columns, feature_importance)
    
    # SHAP分析
    apply_shap_analysis(model, X_test, X.columns, y_test, y_pred_proba)
    
    # 特定特徴量の詳細分析
    analyze_specific_features(model, X_test, y_test, y_pred_proba)
    
    print("\n=== 分析完了 ===")
    print("生成されたファイル:")
    print("- feature_importance.png: 特徴量重要度")
    print("- confusion_matrix.png: 混同行列")
    print("- pdp_ice_plots.png: PDPとICEプロット")
    print("- shap_importance.png: SHAP重要度")
    print("- shap_summary.png: SHAP詳細")

if __name__ == "__main__":
    main() 