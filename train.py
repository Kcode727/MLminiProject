import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import json
import os
from datetime import datetime

DATA_PATH = "Walmart.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

def create_features(df):
    data = df.copy()
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Week'] = data['Date'].dt.isocalendar().week.astype(int)
    data['Quarter'] = data['Date'].dt.quarter
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['IsMonthStart'] = (data['Date'].dt.day <= 7).astype(int)
    data['IsMonthEnd'] = (data['Date'].dt.day >= 23).astype(int)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['Week_sin'] = np.sin(2 * np.pi * data['Week'] / 52)
    data['Week_cos'] = np.cos(2 * np.pi * data['Week'] / 52)

    for store in data['Store'].unique():
        idx = data['Store'] == store
        sales = data.loc[idx, 'Weekly_Sales']
        for lag in [1, 2, 4, 8, 12]:
            data.loc[idx, f'Sales_Lag_{lag}'] = sales.shift(lag)
        for w in [4, 8, 12]:
            data.loc[idx, f'Sales_RollMean_{w}'] = sales.rolling(w, min_periods=1).mean()
            data.loc[idx, f'Sales_RollStd_{w}'] = sales.rolling(w, min_periods=1).std().fillna(0)

    for col in data.columns:
        if 'Lag' in col or 'Roll' in col:
            data[col] = data.groupby('Store')[col].transform(lambda x: x.fillna(x.mean()))

    data['Holiday_Unemp'] = data['Holiday_Flag'] * data['Unemployment']
    data['Holiday_Temp'] = data['Holiday_Flag'] * data['Temperature']
    data['CPI_Unemp'] = data['CPI'] * data['Unemployment']
    data['Week_Counter'] = data.groupby('Store').cumcount() + 1

    return data

df_full = create_features(df)

percentiles = np.percentile(df_full['Weekly_Sales'], [33, 67])
df_full['Sales_Category'] = np.where(
    df_full['Weekly_Sales'] < percentiles[0], 0,
    np.where(df_full['Weekly_Sales'] < percentiles[1], 1, 2)
)

feature_cols = [
    'Store', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag',
    'Year', 'Month', 'Week', 'Quarter', 'DayOfYear', 'IsMonthStart', 'IsMonthEnd',
    'Month_sin', 'Month_cos', 'Week_sin', 'Week_cos',
    'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_4', 'Sales_Lag_8', 'Sales_Lag_12',
    'Sales_RollMean_4', 'Sales_RollMean_8', 'Sales_RollMean_12',
    'Sales_RollStd_4', 'Sales_RollStd_8', 'Sales_RollStd_12',
    'Holiday_Unemp', 'Holiday_Temp', 'CPI_Unemp', 'Week_Counter'
]

X = df_full[feature_cols]
y = df_full['Sales_Category']

split_date = df_full['Date'].quantile(0.8)
train_idx = df_full['Date'] < split_date
X_train, X_val = X[train_idx], X[~train_idx]
y_train, y_val = y[train_idx], y[~train_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("Initializing your 13 models...")

models = {
    'Decision_Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random_Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient_Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'AdaBoost': AdaBoostClassifier(random_state=RANDOM_STATE),
    'Logistic_Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Ridge_Classifier': RidgeClassifier(random_state=RANDOM_STATE),
    'Naive_Bayes': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(),
    'SVM_RBF': SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True),
    'Neural_Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=RANDOM_STATE, max_iter=500),
}

try:
    import xgboost as xgb
    models['XGBoost'] = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss')
    print("XGBoost included.")
except ImportError:
    print("XGBoost not installed. Skipping.")

try:
    import lightgbm as lgb
    models['LightGBM'] = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    print("LightGBM included.")
except ImportError:
    print("LightGBM not installed. Skipping.")


print(f"Training {len(models)} models...")
results = []
for name, model in models.items():
    print(f"  → {name}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    f1 = f1_score(y_val, y_pred, average='weighted')
    results.append({'name': name, 'model': model, 'f1': f1})

results = sorted(results, key=lambda x: x['f1'], reverse=True)

top_n = 3
top3 = results[:top_n]
top3_models = [r['model'] for r in top3]
top3_names = [r['name'] for r in top3]
top3_f1 = [r['f1'] for r in top3]

# Save top 3
joblib.dump(top3_models, f'{MODELS_DIR}/top3_models.pkl')
joblib.dump(top3_names, f'{MODELS_DIR}/top3_names.pkl')
joblib.dump(top3_f1, f'{MODELS_DIR}/top3_f1.pkl')

leaderboard = [
    {'rank': i+1, 'name': r['name'], 'f1': round(r['f1'], 4)}
    for i, r in enumerate(results)
]

# Full metrics
full_metrics = {
    'training_date': datetime.now().isoformat(),
    'total_records': len(df_full),
    'train_size': len(X_train),
    'val_size': len(X_val),
    'percentiles': percentiles.tolist(),
    'feature_count': len(feature_cols),
    'top3_models': top3_names,
    'top3_f1': top3_f1,
    'leaderboard': leaderboard
}

with open(f'{MODELS_DIR}/model_metrics.json', 'w') as f:
    json.dump(full_metrics, f, indent=2)

joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')
joblib.dump(feature_cols, f'{MODELS_DIR}/feature_names.pkl')
joblib.dump(percentiles, f'{MODELS_DIR}/percentiles.pkl')
joblib.dump(df_full, f'{MODELS_DIR}/training_data_full.pkl')

print(f"\nSAVED:")
print(f"  • Top {top_n} models: {top3_names}")
print(f"  • All {len(models)} F1 scores in model_metrics.json")
print(f"\nTOP {top_n}:")
for i, (name, f1) in enumerate(zip(top3_names, top3_f1), 1):
    print(f"  {i}. {name}: F1 = {f1:.4f}")


def prepare_one_row(user_input, df_full, feature_names):
    import pandas as pd
    import numpy as np

    store = user_input['Store']
    target_date = pd.to_datetime(user_input['Date'])
    
    # Get store history
    store_data = df_full[df_full['Store'] == store].copy()
    store_data = store_data.sort_values('Date').reset_index(drop=True)
    
    if store_data.empty:
        raise ValueError("No historical data for this store.")

    # Past data up to target_date
    past_data = store_data[store_data['Date'] < target_date]
    if past_data.empty:
        past_data = store_data 

    sales = past_data['Weekly_Sales']
    mean_sales = sales.mean() if len(sales) > 0 else 0
    std_sales = sales.std() if len(sales) > 1 else 1

    # Normalize base features
    stats = {
        'Temperature': (df_full['Temperature'].mean(), df_full['Temperature'].std()),
        'Fuel_Price': (df_full['Fuel_Price'].mean(), df_full['Fuel_Price'].std()),
        'CPI': (df_full['CPI'].mean(), df_full['CPI'].std()),
        'Unemployment': (df_full['Unemployment'].mean(), df_full['Unemployment'].std()),
    }

    def z_normalize(val, key):
        mean, std = stats[key]
        return (val - mean) / (std + 1e-8)

    features = {
        'Store': float(store),
        'Temperature': z_normalize(user_input['Temperature'], 'Temperature'),
        'Fuel_Price': z_normalize(user_input['Fuel_Price'], 'Fuel_Price'),
        'CPI': z_normalize(user_input['CPI'], 'CPI'),
        'Unemployment': z_normalize(user_input['Unemployment'], 'Unemployment'),
        'Holiday_Flag': float(user_input['Holiday_Flag']),
        'Year': target_date.year,
        'Month': target_date.month,
        'Week': int(target_date.isocalendar().week),
        'Quarter': target_date.quarter,
        'DayOfYear': target_date.dayofyear,
        'IsMonthStart': 1 if target_date.day <= 7 else 0,
        'IsMonthEnd': 1 if target_date.day >= 23 else 0,
        'Month_sin': np.sin(2 * np.pi * target_date.month / 12),
        'Month_cos': np.cos(2 * np.pi * target_date.month / 12),
        'Week_sin': np.sin(2 * np.pi * target_date.isocalendar().week / 52),
        'Week_cos': np.cos(2 * np.pi * target_date.isocalendar().week / 52),
    }

    if len(sales) > 0:
        for lag in [1, 2, 4, 8, 12]:
            val = sales.iloc[-lag] if len(sales) >= lag else mean_sales
            features[f'Sales_Lag_{lag}'] = (val - mean_sales) / (std_sales + 1e-8)
        for w in [4, 8, 12]:
            roll = sales.tail(w)
            features[f'Sales_RollMean_{w}'] = (roll.mean() - mean_sales) / (std_sales + 1e-8) if len(roll) > 0 else 0
            features[f'Sales_RollStd_{w}'] = roll.std() / (std_sales + 1e-8) if len(roll) > 1 else 0
    else:
        for lag in [1, 2, 4, 8, 12]:
            features[f'Sales_Lag_{lag}'] = 0
        for w in [4, 8, 12]:
            features[f'Sales_RollMean_{w}'] = 0
            features[f'Sales_RollStd_{w}'] = 0

    # Interactions
    features['Holiday_Unemp'] = features['Holiday_Flag'] * features['Unemployment']
    features['Holiday_Temp'] = features['Holiday_Flag'] * features['Temperature']
    features['CPI_Unemp'] = features['CPI'] * features['Unemployment']
    features['Week_Counter'] = len(store_data) / 100.0

    # Final DataFrame
    input_df = pd.DataFrame([features])
    input_df = input_df[feature_names]  
    return input_df.values  