import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Walmart Sales Classifier", page_icon="Chart", layout="wide")

# --------------------------------------------------------------
# CSS (unchanged – copy-paste from your original file)
# --------------------------------------------------------------
st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: 800; color: #1f2937; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.1rem; color: #6b7280; text-align: center; margin-bottom: 2rem;}
    .metric-card {border: 1px solid #e5e7eb; padding: 1rem; border-radius: 12px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}
    .metric-card h3 {color: #6a6c6e; font-size: 0.85rem; font-weight: 600; margin:0; text-transform: uppercase; letter-spacing: 1px;}
    .metric-card h2 {margin: 0.5rem 0 0 0; color: #525559; font-size: 1.8rem; font-weight: 700;}
    .glass-card {backdrop-filter: blur(10px); border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08);}
    .prediction-box {padding: 2rem; border-radius: 16px; text-align: center; border: 2px solid #e5e7eb; box-shadow: 0 8px 16px rgba(0,0,0,0.1);}
    .prediction-box h1 {font-size: 3.5rem; margin: 0; font-weight: 800; color: #525559;}
    .prediction-box h3 {color: #6b7280; font-size: 0.9rem; margin: 0 0 0.5rem 0; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px;}
    .section-header {font-size: 1.6rem; font-weight: 700; color: #6a6c6e; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #e5e7eb;}
    .feature-card {border-radius: 8px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .feature-badge {display: inline-block; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem; color: #374151; background: #e5e7eb;}
    .stTabs [data-baseweb="tab"] {border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 600;}
    .stButton>button {border-radius: 8px; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

CHART_CONFIG = {'displayModeBar': False}

# --------------------------------------------------------------
# FEATURE DICTIONARY
# --------------------------------------------------------------
@st.cache_data
def load_feature_dict():
    try:
        with open("feature_dict.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("feature_dict.json not found in main folder!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading feature_dict.json: {e}")
        st.stop()

FEATURE_DICTIONARY = load_feature_dict()

# --------------------------------------------------------------
# MODEL CLASSES (same as before)
# --------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

MODEL_CLASSES = {
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
    MODEL_CLASSES['XGBoost'] = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss')
except ImportError:
    pass

try:
    import lightgbm as lgb
    MODEL_CLASSES['LightGBM'] = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
except ImportError:
    pass

# --------------------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        top3_models = joblib.load('models/top3_models.pkl')
        top3_names = joblib.load('models/top3_names.pkl')
        top3_f1 = joblib.load('models/top3_f1.pkl')

        for name in top3_names:
            if name not in MODEL_CLASSES:
                st.error(f"Model '{name}' trained but not available. Install missing packages.")
                st.stop()

        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        percentiles = joblib.load('models/percentiles.pkl')
        df_full = joblib.load('models/training_data_full.pkl')
        df_raw = pd.read_csv('Walmart.csv')
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], dayfirst=True)
        with open('models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return top3_models, top3_names, top3_f1, scaler, feature_names, percentiles, df_full, df_raw, metrics
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

top3_models, top3_names, top3_f1, scaler, feature_names, percentiles, df_full, df_raw, metrics = load_artifacts()

# --------------------------------------------------------------
# *** prepare_one_row – EMBEDDED DIRECTLY IN app.py ***
# --------------------------------------------------------------
def prepare_one_row(user_input, df_full, feature_names):
    """
    Transform a single user dict into the exact feature vector expected by the trained models.
    """
    import pandas as pd
    import numpy as np

    store = user_input['Store']
    target_date = pd.to_datetime(user_input['Date'])

    # ---- historical data for this store ---------------------------------
    store_data = df_full[df_full['Store'] == store].copy()
    store_data = store_data.sort_values('Date').reset_index(drop=True)

    if store_data.empty:
        raise ValueError("No historical data for this store.")

    past_data = store_data[store_data['Date'] < target_date]
    if past_data.empty:
        past_data = store_data

    sales = past_data['Weekly_Sales']
    mean_sales = sales.mean() if len(sales) > 0 else 0
    std_sales  = sales.std()  if len(sales) > 1 else 1

    # ---- Z-normalisation helpers ----------------------------------------
    stats = {
        'Temperature': (df_full['Temperature'].mean(), df_full['Temperature'].std()),
        'Fuel_Price' : (df_full['Fuel_Price'].mean(),  df_full['Fuel_Price'].std()),
        'CPI'        : (df_full['CPI'].mean(),        df_full['CPI'].std()),
        'Unemployment': (df_full['Unemployment'].mean(), df_full['Unemployment'].std()),
    }

    def z_normalize(val, key):
        mean, std = stats[key]
        return (val - mean) / (std + 1e-8)

    # ---- base features --------------------------------------------------
    features = {
        'Store'        : float(store),
        'Temperature'  : z_normalize(user_input['Temperature'], 'Temperature'),
        'Fuel_Price'   : z_normalize(user_input['Fuel_Price'],  'Fuel_Price'),
        'CPI'          : z_normalize(user_input['CPI'],        'CPI'),
        'Unemployment' : z_normalize(user_input['Unemployment'],'Unemployment'),
        'Holiday_Flag' : float(user_input['Holiday_Flag']),
        'Year'         : target_date.year,
        'Month'        : target_date.month,
        'Week'         : int(target_date.isocalendar().week),
        'Quarter'      : target_date.quarter,
        'DayOfYear'    : target_date.dayofyear,
        'IsMonthStart' : 1 if target_date.day <= 7 else 0,
        'IsMonthEnd'   : 1 if target_date.day >= 23 else 0,
        'Month_sin'    : np.sin(2 * np.pi * target_date.month / 12),
        'Month_cos'    : np.cos(2 * np.pi * target_date.month / 12),
        'Week_sin'     : np.sin(2 * np.pi * target_date.isocalendar().week / 52),
        'Week_cos'     : np.cos(2 * np.pi * target_date.isocalendar().week / 52),
    }

    # ---- lag features ---------------------------------------------------
    if len(sales) > 0:
        for lag in [1, 2, 4, 8, 12]:
            val = sales.iloc[-lag] if len(sales) >= lag else mean_sales
            features[f'Sales_Lag_{lag}'] = (val - mean_sales) / (std_sales + 1e-8)
        for w in [4, 8, 12]:
            roll = sales.tail(w)
            features[f'Sales_RollMean_{w}'] = (roll.mean() - mean_sales) / (std_sales + 1e-8) if len(roll) > 0 else 0
            features[f'Sales_RollStd_{w}']  = roll.std() / (std_sales + 1e-8) if len(roll) > 1 else 0
    else:
        for lag in [1, 2, 4, 8, 12]:
            features[f'Sales_Lag_{lag}'] = 0
        for w in [4, 8, 12]:
            features[f'Sales_RollMean_{w}'] = 0
            features[f'Sales_RollStd_{w}']  = 0

    # ---- interaction features -------------------------------------------
    features['Holiday_Unemp'] = features['Holiday_Flag'] * features['Unemployment']
    features['Holiday_Temp']  = features['Holiday_Flag'] * features['Temperature']
    features['CPI_Unemp']     = features['CPI'] * features['Unemployment']
    features['Week_Counter']  = len(store_data) / 100.0

    # ---- final DataFrame ------------------------------------------------
    input_df = pd.DataFrame([features])
    input_df = input_df[feature_names]               # enforce exact column order
    return input_df.values

# --------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------
if 'dev_mode' not in st.session_state:
    st.session_state.dev_mode = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# --------------------------------------------------------------
# DEVELOPER MODE TOGGLE
# --------------------------------------------------------------
col1, col2, col3 = st.columns([5, 1, 1])
with col2:
    if st.button("Dev Mode" if not st.session_state.dev_mode else "User Mode", type="secondary"):
        st.session_state.dev_mode = not st.session_state.dev_mode
        st.rerun()

# --------------------------------------------------------------
# USER MODE (Dashboard + Predict)
# --------------------------------------------------------------
if not st.session_state.dev_mode:
    tab1, tab2 = st.tabs(["Dashboard", "Predict"])

    # ---------- Dashboard ----------
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Total Records</h3><h2>{len(df_full):,}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Stores</h3><h2>{df_full["Store"].nunique()}</h2></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>Avg Sales</h3><h2>${df_full["Weekly_Sales"].mean():,.0f}</h2></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h3>Models Used</h3><h2>Top 3</h2></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Sales Trends</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            monthly = df_full.groupby(df_full['Date'].dt.to_period('M'))['Weekly_Sales'].mean()
            monthly.index = monthly.index.to_timestamp()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values, mode='lines', line=dict(color='#667eea', width=3), fill='tozeroy'))
            fig.update_layout(title="<b>Monthly Average Sales</b>", height=400)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        with col2:
            counts = df_full['Sales_Category'].value_counts().sort_index()
            labels = ['Low', 'Medium', 'High']
            fig = go.Figure(data=[go.Pie(labels=labels, values=counts.values, hole=0.5,
                                        marker_colors=['#ef4444', '#f59e0b', '#10b981'])])
            fig.update_layout(title="<b>Sales Category Distribution</b>", height=400)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

        st.markdown('<div class="section-header">Store & Holiday Impact</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            store_avg = df_full.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
            fig = go.Figure(go.Bar(x=store_avg.index.astype(str), y=store_avg.values,
                                   marker=dict(color=store_avg.values, colorscale='Blues')))
            fig.update_layout(title="<b>All Stores by Avg Sales</b>", height=450,
                              xaxis_title="Store ID", xaxis=dict(tickangle=-45))
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        with col2:
            holiday = df_full.groupby('Holiday_Flag')['Weekly_Sales'].mean()
            boost = ((holiday[1] - holiday[0]) / holiday[0]) * 100
            fig = go.Figure(go.Bar(x=['Normal', 'Holiday'], y=holiday.values,
                                   marker=dict(color=['#3b82f6', '#10b981'])))
            fig.add_annotation(x=0.5, y=max(holiday.values)*1.15,
                               text=f"<b>+{boost:.1f}%</b>", showarrow=False)
            fig.update_layout(title="<b>Holiday vs Normal</b>", height=450)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    # ---------- Predict ----------
    with tab2:
        st.markdown('<div class="section-header">Predict Sales Category</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            store = st.selectbox("Store ID", sorted(df_full['Store'].unique()))
            date = st.date_input("Date", value=datetime.now())
            holiday = st.selectbox("Holiday Week", ["No", "Yes"])
        with col2:
            temperature = st.number_input("Temperature (°F)", 20.0, 100.0, 70.0, 1.0)
            fuel_price = st.number_input("Fuel Price ($)", 2.0, 5.0, 3.5, 0.01)
        with col3:
            cpi = st.number_input("CPI", 150.0, 250.0, 220.0, 0.1)
            unemployment = st.number_input("Unemployment (%)", 3.0, 15.0, 7.5, 0.1)

        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            predict_button = st.button("Predict", type="primary", use_container_width=True)

        if predict_button:
            with st.spinner("Top 3 models voting..."):
                input_dict = {
                    'Store': store,
                    'Date': pd.to_datetime(date),
                    'Temperature': temperature,
                    'Fuel_Price': fuel_price,
                    'CPI': cpi,
                    'Unemployment': unemployment,
                    'Holiday_Flag': 1 if holiday == "Yes" else 0
                }
                try:
                    X_input = prepare_one_row(input_dict, df_full, feature_names)
                    votes = []
                    confidences = []
                    for model in top3_models:
                        pred = model.predict(X_input)[0]
                        proba = model.predict_proba(X_input)[0]
                        votes.append(pred)
                        confidences.append(proba[pred])

                    final_class = pd.Series(votes).mode()[0]
                    avg_confidence = np.mean(confidences) * 100
                    agreement = sum(1 for v in votes if v == final_class)

                    category = ['Low', 'Medium', 'High'][final_class]
                    sales_range = (
                        f"< ${percentiles[0]:,.0f}" if final_class == 0 else
                        f"${percentiles[0]:,.0f} – ${percentiles[1]:,.0f}" if final_class == 1 else
                        f"> ${percentiles[1]:,.0f}"
                    )

                    st.session_state.prediction = {
                        'category': category,
                        'confidence': avg_confidence,
                        'range': sales_range,
                        'agreement': f"{agreement}/3",
                        'votes': dict(zip(top3_names, [(['Low','Medium','High'][v]) for v in votes])),
                        'confidences': dict(zip(top3_names, [f"{c*100:.1f}%" for c in confidences]))
                    }
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        if st.session_state.prediction:
            pred = st.session_state.prediction
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="prediction-box"><h3>PREDICTION</h3><h1>{pred["category"]}</h1></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="glass-card" style="text-align:center;"><h3>CONFIDENCE</h3><h2>{pred["confidence"]:.1f}%</h2><p style="margin:0; color:#6b7280;">{pred["agreement"]} agree</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="glass-card" style="text-align:center;"><h3>RANGE</h3><h2>{pred["range"]}</h2></div>', unsafe_allow_html=True)

            st.markdown("### Model Votes")
            vote_df = pd.DataFrame({
                'Model': top3_names,
                'Vote': [pred['votes'][n] for n in top3_names],
                'Confidence': [pred['confidences'][n] for n in top3_names]
            })
            st.table(vote_df)

# --------------------------------------------------------------
# DEVELOPER MODE
# --------------------------------------------------------------
else:
    st.markdown("## Developer Mode")
    tabs = st.tabs(["Feature Dictionary", "Model Leaderboard"])

    with tabs[0]:
        st.markdown("### Feature Dictionary & Visualizations")
        for name, info in FEATURE_DICTIONARY.items():
            with st.expander(f"**{name}**", expanded=False):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"""
                    <div class="feature-card">
                        <span class="feature-badge">{info['category']}</span>
                        <h4>{name}</h4>
                        <p><strong>Why:</strong> {info['why']}</p>
                        <p><strong>Example:</strong><br><code>{info['example']}</code></p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    plot_type = info.get("plot_type", "none")
                    if plot_type == "cyclical":
                        period = info["plot_data"]["period"]
                        labels = info["plot_data"].get("labels", [])
                        x = np.linspace(0, 2 * np.pi, 100)
                        sin_wave = np.sin(x * period / (2 * np.pi))
                        cos_wave = np.cos(x * period / (2 * np.pi))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=sin_wave, name="sin", line=dict(color="#667eea")))
                        fig.add_trace(go.Scatter(x=x, y=cos_wave, name="cos", line=dict(color="#f59e0b")))
                        fig.update_layout(height=200, margin=dict(l=0,r=0,t=30,b=0),
                                          xaxis_title="Cycle", yaxis=dict(range=[-1.2, 1.2]))
                        if labels:
                            tickvals = np.linspace(0, 2 * np.pi, period + 1)[:-1]
                            fig.update_xaxes(tickvals=tickvals, ticktext=labels)
                        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

                    elif plot_type == "bar_boost":
                        feat = info["plot_data"]["feature"]
                        if feat in df_full.columns:
                            boost = df_full.groupby(feat)['Weekly_Sales'].mean()
                            boost_pct = ((boost[1] - boost[0]) / boost[0] * 100) if len(boost) > 1 else 0
                            fig = go.Figure(go.Bar(
                                x=['No', 'Yes'],
                                y=boost.values,
                                marker_color=['#94a3b8', '#10b981']
                            ))
                            fig.add_annotation(x=1, y=boost.values[-1]*1.1,
                                               text=f"+{boost_pct:.1f}%", showarrow=False,
                                               font=dict(size=14, color="#10b981"))
                            fig.update_layout(height=200, margin=dict(l=0,r=0,t=30,b=0), title=f"{feat} Impact")
                            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

                    elif plot_type == "scatter_lag":
                        lag = info["plot_data"]["lag"]
                        store_sample = df_full[df_full['Store'] == df_full['Store'].iloc[0]].sort_values('Date')
                        if len(store_sample) > lag:
                            x_lag = store_sample['Weekly_Sales'].shift(lag).iloc[lag:lag+50]
                            y_actual = store_sample['Weekly_Sales'].iloc[lag:lag+50]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=x_lag, y=y_actual, mode='markers',
                                                    marker=dict(color='#667eea', size=6)))
                            fig.add_trace(go.Scatter(x=[0, 2e6], y=[0, 2e6], mode='lines',
                                                    line=dict(dash='dash', color='#1f2937')))
                            fig.update_layout(height=200, margin=dict(l=0,r=0,t=30,b=0),
                                              xaxis_title=f"Lag {lag}", yaxis_title="Actual Sales")
                            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

                    elif plot_type == "rolling_demo":
                        window = info["plot_data"]["window"]
                        store_sample = df_full[df_full['Store'] == df_full['Store'].iloc[0]].sort_values('Date')
                        if len(store_sample) > window:
                            dates = store_sample['Date'].iloc[:50]
                            actual = store_sample['Weekly_Sales'].iloc[:50]
                            roll = actual.rolling(window, min_periods=1).mean()
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual",
                                                    line=dict(color="#94a3b8")))
                            fig.add_trace(go.Scatter(x=dates, y=roll.iloc[:50],
                                                    name=f"RollMean {window}",
                                                    line=dict(color="#f59e0b", width=3)))
                            fig.update_layout(height=200, margin=dict(l=0,r=0,t=30,b=0))
                            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    with tabs[1]:
        st.markdown("### Full Model Leaderboard")
        leaderboard = pd.DataFrame(metrics['leaderboard'])
        leaderboard = leaderboard[['rank', 'name', 'f1']].copy()
        leaderboard.columns = ['Rank', 'Model', 'F1 Score']
        leaderboard['F1 Score'] = leaderboard['F1 Score'].round(4)
        st.table(leaderboard)

        st.markdown("### Top 3 Models Used")
        top3_df = pd.DataFrame({
            'Rank': [1, 2, 3],
            'Model': top3_names,
            'F1 Score': [f"{f:.4f}" for f in top3_f1]
        })
        st.table(top3_df)