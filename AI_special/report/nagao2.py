import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import PartialDependenceDisplay
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

feature_names = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring',
    'veil-type', 'veil-color', 'ring-number', 'ring-type',
    'spore-print-color', 'population', 'habitat'
]
columns = ['class'] + feature_names

try:
    df = pd.read_csv('agaricus-lepiota.data', header=None, names=columns)
except FileNotFoundError:
    print("データが見つかりません。サンプルデータを使用します。")
    sample_data = [
        ['p','x','s','n','t','p','f','c','n','k','e','e','s','s','w','w','p','w','o','p','k','s','u'],
        ['e','x','s','y','t','a','f','c','b','k','e','c','s','s','w','w','p','w','o','p','n','n','g'],
        ['e','b','s','w','t','l','f','c','b','n','e','c','s','s','w','w','p','w','o','p','n','n','m'],
        ['p','x','y','w','t','p','f','c','n','n','e','e','s','s','w','w','p','w','o','p','k','s','u'],
        ['e','x','s','g','f','n','f','w','b','k','t','e','s','s','w','w','p','w','o','e','n','a','g']
    ]
    np.random.seed(42)
    extended_data = []
    for _ in range(1000):
        base = sample_data[np.random.randint(len(sample_data))].copy()
        for i in range(1, len(base)):
            if np.random.random() < 0.3:
                current_values = list(set([row[i] for row in sample_data]))
                base[i] = np.random.choice(current_values)
        extended_data.append(base)
    df = pd.DataFrame(extended_data, columns=columns)

df.replace('?', np.nan, inplace=True)
for col in df.columns[df.isnull().any()]:
    df[col].fillna(df[col].mode()[0], inplace=True)

label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, verbose=0)
cat_model.fit(X_train, y_train)

y_pred = cat_model.predict(X_test)
print(f"\n精度: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

odor_index = X.columns.get_loc('odor')
PartialDependenceDisplay.from_estimator(
    cat_model, X_test, features=[odor_index], feature_names=X.columns, kind="both"
)
plt.title("PDP & ICE for 'odor'")
plt.tight_layout()
plt.show()

explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(X_test)

print("\nSHAP summary plot (bar)")
shap.summary_plot(shap_values, X_test, plot_type="bar")

print("\nSHAP summary plot (dot)")
shap.summary_plot(shap_values, X_test)
