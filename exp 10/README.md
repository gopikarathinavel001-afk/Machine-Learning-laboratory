# Customer Purchase Prediction — Machine Learning System

A complete, working end-to-end Machine Learning laboratory project that predicts whether a customer visiting an e-commerce website is likely to purchase a product based on their browsing engagement and shopping profile.

---

## 1. Problem Statement

In modern e-commerce platforms, understanding customer buying intent is vital for delivering targeted promotions, optimizing ad spend, and personalizing user experiences. The goal of this project is to build an automated classification system that analyzes customer demographic and real-time shopping behavioral metrics to determine whether the customer will complete a purchase (`Purchase: YES` or `Purchase: NO`).

---

## 2. Project Objectives

- Develop a binary classification system using **Logistic Regression** from Scikit-Learn.
- Demonstrate the complete, standardized Machine Learning workflow:
  1. **Dataset Overview**: Ingestion and target distribution inspection.
  2. **Data Preprocessing**: Handling missing values, target encoding, and feature standardization.
  3. **Exploratory Data Analysis (EDA)**: Visualizing behavioral correlations and feature distributions.
  4. **Model Training**: Pipeline integration (`StandardScaler` + `LogisticRegression`) with an 80/20 stratified split.
  5. **Live Prediction**: Interactive web interface computing purchase outcome and probability (`predict_proba`).
  6. **Model Evaluation**: Assessing model performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
  7. **Prediction History**: Persistent local logging using SQLite3.
- Provide a clean, dark-themed, screenshot-friendly web application designed for academic laboratory records and presentations.

---

## 3. Machine Learning Type

- **Problem Type:** Supervised Learning — Binary Classification
- **Target Encoding:**
  - `0` = No Purchase (`Purchase: NO`)
  - `1` = Purchase Completed (`Purchase: YES`)
- **Estimator:** Logistic Regression (`sklearn.linear_model.LogisticRegression`) with Standard Scaler Pipeline.

---

## 4. Features & Target Variable

### Input Features:

| # | Feature Name | Description | Data Type | Example Value |
|---|--------------|-------------|-----------|---------------|
| 1 | `Age` | Customer age in years | Integer | 21 |
| 2 | `Previous_Purchases` | Total number of prior orders | Integer | 3 |
| 3 | `Website_Time_Minutes` | Time spent browsing the site | Float (Minutes) | 12.0 |
| 4 | `Products_Viewed` | Number of product pages visited | Integer | 8 |
| 5 | `Previous_Spending` | Historical spending amount | Float / Integer (₹) | ₹2,500 |
| 6 | `Discount_Percent` | Promotional discount offered | Float / Integer (%) | 20% |

### Target Variable:

- **Target:** `Purchase`
- **Classes:** `Yes` (1) / `No` (0)

---

## 5. Technologies Used

- **Language:** Python 3.10+
- **Web Framework:** Flask 3.x
- **Frontend:** HTML5, CSS3 (Custom Dark Academic Theme), Vanilla JavaScript (Fetch API)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Logistic Regression, StandardScaler, Pipeline, Metrics)
- **Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib
- **Database:** SQLite3 (Local file-based database)
- **Development Environment:** Visual Studio Code

---

## 6. Project Structure

```
customer purchase prediction/
│
├── app.py                      # Flask web application & API routing
├── generate_data_and_train.py  # Dataset generation, pipeline training & plot generator
├── test_app.py                 # Automated unit and integration test suite
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & laboratory guide
│
├── data/
│   └── customer_purchase_data.csv   # Local dataset (200 records)
│
├── models/
│   └── purchase_model.pkl           # Trained StandardScaler + LogisticRegression pipeline
│
├── static/
│   ├── style.css               # Dark theme responsive stylesheet
│   └── plots/                  # Visualizations generated from actual data
│       ├── class_distribution.png
│       ├── website_time_vs_purchase.png
│       ├── previous_purchases_vs_purchase.png
│       ├── products_viewed_vs_purchase.png
│       ├── previous_spending_vs_purchase.png
│       ├── discount_vs_purchase.png
│       └── confusion_matrix.png
│
├── templates/
│   └── index.html              # Screenshot-friendly single-page laboratory UI
│
└── database/
    └── predictions.db          # SQLite persistent history database
```

---

## 7. Dataset Description

The dataset `data/customer_purchase_data.csv` contains **200 customer records** generated with a reproducible random seed (`random_state=42`). 

- Natural probabilistic relationships are modeled using a logistic function with realistic variance:
  - Customers with higher website duration, more previous purchases, higher historical spending, and greater promotional discounts exhibit higher purchase probabilities.
  - The dataset includes non-linear real-world noise to prevent artificial 100% deterministic relationships.
- Target class balance:
  - `Purchase = Yes`: **94 records (47.0%)**
  - `Purchase = No`: **106 records (53.0%)**

---

## 8. Machine Learning Workflow Steps

### Step 1 — Dataset Overview
Displays total records count, input feature list, first 8 rows of customer data, and target class distribution bar chart.

### Step 2 — Data Preprocessing
- **Missing Values Audit:** 0 missing values detected across all columns.
- **Separation:** Features extracted into $X \in \mathbb{R}^{200 \times 6}$ and target into $y \in \{0, 1\}^{200}$.
- **Feature Scaling:** Uses `StandardScaler` to transform features such that $\mu = 0$ and $\sigma = 1$:
  $$z = \frac{x - \mu}{\sigma}$$
- Displays "✓ Preprocessing Completed Successfully".

### Step 3 — Exploratory Data Analysis (EDA)
Generates 5 distinct high-contrast charts comparing purchasing vs. non-purchasing customers:
1. `Website Time vs Purchase` (Box Plot)
2. `Previous Purchases vs Purchase` (Bar Plot)
3. `Products Viewed vs Purchase` (Box Plot)
4. `Previous Spending vs Purchase` (Box Plot)
5. `Discount Offered vs Purchase` (Bar Chart)

### Step 4 — Model Training
- **Algorithm:** Logistic Regression with L-BFGS solver (`random_state=42`).
- **Dataset Partition:** Stratified 80% Training ($N=160$), 20% Testing ($N=40$).
- **Pipeline:** Fitted `Pipeline(StandardScaler(), LogisticRegression())` serialized to `models/purchase_model.pkl`.
- Displays "✓ Model Training Completed Successfully".

### Step 5 — Customer Purchase Prediction
- Interactive form with real-time validation for all 6 features.
- Quick presets for instant demonstration (Lab Spec Example, High Intent Buyer, Casual Visitor).
- Computes prediction (`Purchase: YES` / `Purchase: NO`) and exact probability using `pipeline.predict_proba()`.

### Step 6 — Model Evaluation
Evaluates the model on the unseen test set ($N=40$) without any hard-coded values:
- **Accuracy:** `85.00%`
- **Precision:** `84.21%`
- **Recall:** `84.21%`
- **F1-Score:** `84.21%`
- **Confusion Matrix:** $TN = 18, FP = 3, FN = 3, TP = 16$ with rendered heatmap.

### Step 7 — Prediction History
Every prediction is automatically saved to `database/predictions.db` in the `prediction_history` table with timestamp and probability. History persists across application restarts and can be managed or cleared on demand.

---

## 9. Laboratory Record Screenshot Guide

For laboratory documentation and project reports, capture the following screenshots:

| Figure Label | Laboratory Section | Description |
|---|---|---|
| **Fig 1** | VS Code Environment | Project explorer showing file tree and workspace layout |
| **Fig 2** | Step 1 — Dataset Overview | Dataset summary cards, first 8 rows table, class distribution plot |
| **Fig 3** | Step 2 — Data Preprocessing | Missing values verification table, scaling parameters, success banner |
| **Fig 4** | Step 3 — EDA Visualizations | Grid of 5 comparative charts with behavioral insights |
| **Fig 5** | Step 4 — Model Training | Logistic Regression parameters, train/test split, training success banner |
| **Fig 6** | Step 5 — Customer Input Form | Input fields filled with customer parameters and preset buttons |
| **Fig 7** | Step 5 — Purchase Prediction Result | Prediction badge (`YES`/`NO`), probability meter, and explanation |
| **Fig 8** | Step 6 — Evaluation Metrics | Accuracy, Precision, Recall, F1-Score summary cards |
| **Fig 9** | Step 6 — Confusion Matrix | Rendered test set confusion matrix heatmap (TN, FP, FN, TP) |
| **Fig 10** | Step 7 — Prediction History | SQLite table displaying logged predictions with timestamps |

---

## 10. Installation & Setup

### Prerequisites:
- Python 3.10 or higher
- Visual Studio Code

### Step 1: Clone or Open Workspace
Open the project folder in Visual Studio Code:
```bash
cd "customer purchase prediction"
```

### Step 2: (Optional) Create and Activate Virtual Environment
```bash
# Windows PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 11. How to Run the Project

### 1. (Optional) Re-generate Dataset & Train Model:
```bash
python generate_data_and_train.py
```

### 2. Launch the Flask Web Application:
```bash
python app.py
```

### 3. Open in Browser:
Navigate to:
```
http://127.0.0.1:5000
```

### 4. Run Automated Unit Tests:
```bash
python test_app.py
```

---

## 12. Final Laboratory Conclusion

> *"The Customer Purchase Prediction system was successfully implemented using Machine Learning. The system preprocesses customer shopping data, performs exploratory data analysis, trains a Logistic Regression classification model, evaluates the model using Accuracy, Precision, Recall and F1-Score, predicts whether a customer is likely to purchase a product, and stores prediction history using SQLite."*

---

## 13. Future Enhancements

- Support for multi-class propensity tiers (e.g., High, Medium, Low purchase probability).
- Integration of ensemble algorithms such as Random Forests or Gradient Boosting (XGBoost) for benchmark comparison.
- Real-time clickstream event tracking via WebSocket integration.
