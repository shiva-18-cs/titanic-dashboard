import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Titanic Dashboard", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Data Analysis Dashboard")
st.markdown("An end-to-end data cleaning, visualization, and ML project on the Titanic dataset.")

# ── Sidebar navigation ────────────────────────────────────────────────────────
section = st.sidebar.radio(
    "Navigate",
    ["📋 Raw Data", "🧹 Data Cleaning", "📊 Visualizations", "🤖 ML Models"]
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = sns.load_dataset("titanic")
    return df

df_raw = load_data()

# ── 1. RAW DATA ───────────────────────────────────────────────────────────────
if section == "📋 Raw Data":
    st.header("📋 Raw Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passengers", df_raw.shape[0])
    col2.metric("Total Columns", df_raw.shape[1])
    col3.metric("Survivors", int(df_raw['survived'].sum()))

    st.subheader("First 10 rows")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Dataset Info")
    info_df = pd.DataFrame({
        "Column": df_raw.columns,
        "Non-Null Count": [df_raw[c].notna().sum() for c in df_raw.columns],
        "Null Count": [df_raw[c].isna().sum() for c in df_raw.columns],
        "Dtype": [str(df_raw[c].dtype) for c in df_raw.columns]
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df_raw.describe(), use_container_width=True)

# ── 2. DATA CLEANING ──────────────────────────────────────────────────────────
elif section == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning Steps")

    st.subheader("Step 1 — Missing Values (Before Cleaning)")
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    missing["% Missing"] = (missing["Missing Values"] / len(df_raw) * 100).round(2)
    st.dataframe(missing[missing["Missing Values"] > 0], use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(df_raw.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Value Heatmap (Before Cleaning)")
    st.pyplot(fig)
    plt.close()

    st.subheader("Step 2 — Drop Redundant Columns")
    drop_cols = ['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive']
    st.code(f"df.drop({drop_cols}, axis=1, inplace=True)")
    st.info("These columns are either duplicates of other columns or have too many missing values (e.g. deck: 77% missing).")

    df_clean = df_raw.drop(drop_cols, axis=1)

    st.subheader("Step 3 — Fill Missing Values")
    st.code("""df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)""")
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])

    st.subheader("Step 4 — Encode Categorical Variables")
    st.code("""le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])        # male=1, female=0
df['embarked'] = le.fit_transform(df['embarked'])""")
    le = LabelEncoder()
    df_clean['sex'] = le.fit_transform(df_clean['sex'])
    df_clean['embarked'] = le.fit_transform(df_clean['embarked'])

    st.subheader("Cleaned Dataset")
    st.dataframe(df_clean.head(10), use_container_width=True)

    missing_after = df_clean.isnull().sum().sum()
    st.success(f"✅ After cleaning: {missing_after} missing values remaining | Shape: {df_clean.shape}")

# ── 3. VISUALIZATIONS ─────────────────────────────────────────────────────────
elif section == "📊 Visualizations":
    st.header("📊 Exploratory Data Analysis")

    # prepare clean df for plots
    df_viz = df_raw.copy()

    col1, col2 = st.columns(2)

    # Survival count
    with col1:
        st.subheader("Survival Count")
        fig, ax = plt.subplots()
        sns.countplot(x='survived', data=df_viz, palette='Set2', ax=ax)
        ax.set_xticklabels(['Not Survived', 'Survived'])
        ax.set_title("Survival Distribution")
        st.pyplot(fig)
        plt.close()

    # Survival by Gender
    with col2:
        st.subheader("Survival by Gender")
        fig, ax = plt.subplots()
        sns.countplot(x='sex', hue='survived', data=df_viz, palette='Set1', ax=ax)
        ax.set_title("Survival by Gender")
        ax.legend(title='Survived', labels=['No', 'Yes'])
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    # Survival by Class
    with col3:
        st.subheader("Survival by Passenger Class")
        fig, ax = plt.subplots()
        sns.countplot(x='pclass', hue='survived', data=df_viz, palette='muted', ax=ax)
        ax.set_title("Survival by Pclass")
        ax.legend(title='Survived', labels=['No', 'Yes'])
        st.pyplot(fig)
        plt.close()

    # Age Distribution
    with col4:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df_viz['age'].dropna(), kde=True, color='steelblue', ax=ax)
        ax.set_title("Age Distribution of Passengers")
        st.pyplot(fig)
        plt.close()

    col5, col6 = st.columns(2)

    # Fare Distribution
    with col5:
        st.subheader("Fare Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df_viz['fare'], kde=True, color='salmon', ax=ax)
        ax.set_title("Fare Distribution")
        st.pyplot(fig)
        plt.close()

    # Survival by Embarkation
    with col6:
        st.subheader("Survival by Embarkation Port")
        fig, ax = plt.subplots()
        sns.countplot(x='embarked', hue='survived', data=df_viz, palette='coolwarm', ax=ax)
        ax.set_title("Survival by Embarked Port")
        ax.legend(title='Survived', labels=['No', 'Yes'])
        st.pyplot(fig)
        plt.close()

    # Age vs Fare scatter
    st.subheader("Age vs Fare (by Survival)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x='age', y='fare', hue='survived', data=df_viz, palette='Set1', alpha=0.6, ax=ax)
    ax.set_title("Age vs Fare coloured by Survival")
    ax.legend(title='Survived', labels=['No', 'Yes'])
    st.pyplot(fig)
    plt.close()

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    df_corr = df_viz.drop(['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive'], axis=1)
    df_corr['sex'] = LabelEncoder().fit_transform(df_corr['sex'])
    df_corr['embarked'] = df_corr['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df_corr['age'] = df_corr['age'].fillna(df_corr['age'].median())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_corr.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)
    plt.close()

# ── 4. ML MODELS ──────────────────────────────────────────────────────────────
elif section == "🤖 ML Models":
    st.header("🤖 Machine Learning Models")

    # Prepare data
    @st.cache_data
    def prepare_ml_data():
        df = sns.load_dataset("titanic")
        df.drop(['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive'], axis=1, inplace=True)
        df['age'] = df['age'].fillna(df['age'].median())
        df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
        le = LabelEncoder()
        df['sex'] = le.fit_transform(df['sex'])
        df['embarked'] = le.fit_transform(df['embarked'])
        X = df.drop('survived', axis=1)
        y = df['survived']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    x_train, x_test, y_train, y_test = prepare_ml_data()
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    model_choice = st.selectbox(
        "Select a Model",
        ["Logistic Regression", "Support Vector Machine", "Naive Bayes", "Decision Tree"]
    )

    model_map = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier()
    }

    model = model_map[model_choice]
    model.fit(x_train_s, y_train)
    y_pred = model.predict(x_test_s)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.subheader(f"Results: {model_choice}")
    st.metric("Accuracy", f"{acc*100:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Survived', 'Survived'],
                    yticklabels=['Not Survived', 'Survived'])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df, use_container_width=True)

    st.subheader("Model Accuracy Comparison")
    results = {}
    for name, m in model_map.items():
        m.fit(x_train_s, y_train)
        results[name] = round(accuracy_score(y_test, m.predict(x_test_s)) * 100, 2)

    fig, ax = plt.subplots()
    bars = ax.barh(list(results.keys()), list(results.values()), color=['#4C72B0','#DD8452','#55A868','#C44E52'])
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(60, 100)
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f"{val}%", va='center')
    ax.set_title("All Models — Accuracy Comparison")
    st.pyplot(fig)
    plt.close()
