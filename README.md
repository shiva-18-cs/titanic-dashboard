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

## 🤝 Contributing

Contributions are welcome and appreciated! Here are some ways you can help improve this project:

### 🐛 Reporting Bugs

If you find a bug, please [open an issue](https://github.com/your-username/titanic-dashboard/issues) and include:

- A clear description of the problem
- Steps to reproduce it
- Your Python version and OS
- Any error messages or screenshots

### 💡 Suggesting Features

Have an idea? Open a feature request issue with:

- A clear description of what you'd like to see
- Why it would be useful
- Any relevant examples or references

### 🔧 Submitting a Pull Request

1. **Fork** the repository
2. **Create a branch** for your change:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and test them locally:
   ```bash
   streamlit run app.py
   ```
4. **Commit** with a descriptive message:
   ```bash
   git commit -m "feat: add ROC curve plot to ML section"
   ```
5. **Push** your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** against `main` and describe what you changed and why.

### 🌱 Ideas for Contributions

Here are some good starting points if you're looking for something to work on:

| Area | Ideas |
|---|---|
| **Visualizations** | Add box plots, violin plots, pair plots, or survival rate by age group |
| **ML Models** | Add Random Forest, XGBoost, or k-NN; add ROC/AUC curves |
| **Hyperparameter Tuning** | Add sliders in the sidebar to tune model parameters interactively |
| **Feature Engineering** | Add a title extraction feature from passenger names |
| **Export** | Allow users to download cleaned data or model results as CSV |
| **Tests** | Add unit tests for the data cleaning and preprocessing steps |
| **Deployment** | Add a `Dockerfile` or a guide for deploying to Streamlit Cloud |

### 📐 Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code
- Keep Streamlit sections clearly separated with comments
- Cache expensive operations with `@st.cache_data`
- Close all matplotlib figures with `plt.close()` after rendering

### 💬 Questions?

Feel free to open a discussion or reach out via [GitHub Issues](https://github.com/your-username/titanic-dashboard/issues). All skill levels are welcome — first-time contributors especially encouraged!

---

## 📄 License

MIT License. Feel free to use and modify.
