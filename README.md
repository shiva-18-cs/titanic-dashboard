# titanic-dashboard
# 🚢 Titanic Data Analysis Dashboard

An end-to-end interactive data science project built with **Streamlit**, covering data exploration, cleaning, visualization, and machine learning on the classic Titanic dataset.

---

## 📌 Features

- **Raw Data Explorer** — View the dataset, column info, null counts, and statistical summary
- **Data Cleaning Pipeline** — Step-by-step walkthrough of missing value handling, column dropping, and encoding
- **Exploratory Visualizations** — Charts for survival by gender, class, age, fare, embarkation port, and a correlation heatmap
- **ML Model Comparison** — Train and evaluate four classifiers with accuracy metrics, confusion matrix, and classification report

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `pandas` / `numpy` | Data manipulation |
| `seaborn` / `matplotlib` | Visualization |
| `scikit-learn` | Machine learning |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/titanic-dashboard.git
cd titanic-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📦 Requirements

```
streamlit
numpy
pandas
seaborn
matplotlib
scikit-learn
```

Or generate via:

```bash
pip freeze > requirements.txt
```

---

## 📂 Project Structure

```
titanic-dashboard/
├── app.py          # Main Streamlit application
├── requirements.txt
└── README.md
```

---

## 🧹 Data Cleaning Summary

| Step | Action |
|---|---|
| Drop columns | `class`, `who`, `adult_male`, `deck`, `embark_town`, `alive` |
| Fill `age` | Replaced nulls with **median age** |
| Fill `embarked` | Replaced nulls with **mode** |
| Encode `sex` | Label encoded (`female=0`, `male=1`) |
| Encode `embarked` | Label encoded |

---

## 🤖 ML Models & Results

Four classifiers are trained on an 80/20 train-test split with `StandardScaler` preprocessing:

- Logistic Regression
- Support Vector Machine (SVC)
- Naive Bayes
- Decision Tree Classifier

Results (accuracy) are displayed both individually and as a side-by-side bar chart comparison.

---

## 📊 Dashboard Sections

| Section | Description |
|---|---|
| 📋 Raw Data | Shape, dtypes, nulls, descriptive stats |
| 🧹 Data Cleaning | Step-by-step cleaning with code snippets |
| 📊 Visualizations | 7 plots covering key survival factors |
| 🤖 ML Models | Model selection, metrics, and comparison |

---

## 📄 License

MIT License. Feel free to use and modify.
