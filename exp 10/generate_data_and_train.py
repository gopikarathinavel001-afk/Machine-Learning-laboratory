import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Set styling for matplotlib
plt.style.use('dark_background')
DARK_BG = '#1e293b'
ACCENT_BLUE = '#38bdf8'
ACCENT_GREEN = '#34d399'
ACCENT_RED = '#f87171'
TEXT_COLOR = '#f1f5f9'
GRID_COLOR = '#334155'

def generate_dataset(n_samples=200, random_seed=42):
    """
    Generates a realistic customer purchase dataset with reproducible seed.
    """
    np.random.seed(random_seed)
    
    # 1. Age (18 to 65)
    age = np.random.randint(18, 66, size=n_samples)
    
    # 2. Previous Purchases (0 to 12)
    previous_purchases = np.random.choice(
        np.arange(0, 13), 
        size=n_samples, 
        p=[0.15, 0.18, 0.16, 0.14, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01]
    )
    
    # 3. Website Time in minutes (1.5 to 35.0 mins)
    website_time = np.round(np.random.gamma(shape=3.0, scale=3.5, size=n_samples) + 1.5, 1)
    website_time = np.clip(website_time, 1.0, 35.0)
    
    # 4. Products Viewed (1 to 25)
    products_viewed = np.clip(np.random.poisson(lam=website_time * 0.5 + 2), 1, 25)
    
    # 5. Previous Spending in INR (0 to 12,000)
    previous_spending = np.round(previous_purchases * np.random.uniform(500, 1200, size=n_samples) + np.random.uniform(0, 500, size=n_samples), -1)
    previous_spending = np.clip(previous_spending, 0, 15000)
    
    # 6. Discount Offered in % (0 to 50)
    discount_percent = np.random.choice([0, 5, 10, 15, 20, 25, 30, 40, 50], size=n_samples, p=[0.12, 0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.05, 0.03])
    
    # Calculate logit with realistic weights and natural noise
    # Standardize internally for clean probabilistic contribution
    logit = (
        - 3.2
        + 0.015 * (age - 35)
        + 0.32 * previous_purchases
        + 0.16 * (website_time - 10)
        + 0.14 * (products_viewed - 6)
        + 0.00042 * (previous_spending - 2000)
        + 0.055 * discount_percent
        + np.random.normal(0, 0.75, size=n_samples)
    )
    
    prob = 1.0 / (1.0 + np.exp(-logit))
    purchase_numeric = (prob >= 0.5).astype(int)
    purchase = np.where(purchase_numeric == 1, 'Yes', 'No')
    
    df = pd.DataFrame({
        'Age': age,
        'Previous_Purchases': previous_purchases,
        'Website_Time_Minutes': website_time,
        'Products_Viewed': products_viewed,
        'Previous_Spending': previous_spending,
        'Discount_Percent': discount_percent,
        'Purchase': purchase
    })
    
    return df

def save_plots(df, y_test=None, y_pred=None, plots_dir='static/plots'):
    """
    Generates high-contrast, beautiful dark-themed plots for EDA and Model Evaluation.
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    def apply_dark_theme(fig, ax):
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=10)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, linestyle='--', alpha=0.3, color=GRID_COLOR)

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['Purchase'].value_counts()
    colors = [ACCENT_GREEN if k == 'Yes' else ACCENT_RED for k in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor=TEXT_COLOR, linewidth=0.8)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{int(yval)} ({yval/len(df)*100:.1f}%)', 
                ha='center', va='bottom', color=TEXT_COLOR, fontweight='bold', fontsize=11)
    ax.set_title('Step 1: Target Class Distribution (Purchase Yes vs No)', fontsize=12, pad=12, fontweight='bold')
    ax.set_xlabel('Purchase Decision', fontsize=11)
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.set_ylim(0, max(counts.values) + 18)
    apply_dark_theme(fig, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    # 2. Website Time vs Purchase (Box Plot)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Purchase', y='Website_Time_Minutes', hue='Purchase', data=df, palette={'Yes': ACCENT_GREEN, 'No': ACCENT_RED}, legend=False, ax=ax, width=0.45)
    ax.set_title('EDA: Website Time vs Purchase Decision', fontsize=12, pad=12, fontweight='bold')
    ax.set_xlabel('Purchase Decision', fontsize=11)
    ax.set_ylabel('Website Time (Minutes)', fontsize=11)
    apply_dark_theme(fig, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'website_time_vs_purchase.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    # 3. Previous Purchases vs Purchase (Bar / Count Plot)
    fig, ax = plt.subplots(figsize=(6, 4))
    avg_prev = df.groupby('Purchase')['Previous_Purchases'].mean()
    bars = ax.bar(avg_prev.index, avg_prev.values, color=[ACCENT_RED, ACCENT_GREEN], width=0.45, edgecolor=TEXT_COLOR)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f} avg', 
                ha='center', va='bottom', color=TEXT_COLOR, fontweight='bold', fontsize=11)
    ax.set_title('EDA: Average Previous Purchases vs Purchase', fontsize=12, pad=12, fontweight='bold')
    ax.set_xlabel('Purchase Decision', fontsize=11)
    ax.set_ylabel('Avg Previous Purchases Count', fontsize=11)
    ax.set_ylim(0, max(avg_prev.values) + 1.2)
    apply_dark_theme(fig, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'previous_purchases_vs_purchase.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    # 4. Products Viewed vs Purchase (Box Plot)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Purchase', y='Products_Viewed', hue='Purchase', data=df, palette={'Yes': ACCENT_GREEN, 'No': ACCENT_RED}, legend=False, ax=ax, width=0.45)
    ax.set_title('EDA: Products Viewed vs Purchase Decision', fontsize=12, pad=12, fontweight='bold')
    ax.set_xlabel('Purchase Decision', fontsize=11)
    ax.set_ylabel('Number of Products Viewed', fontsize=11)
    apply_dark_theme(fig, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'products_viewed_vs_purchase.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    # 5. Previous Spending vs Purchase (Box Plot)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Purchase', y='Previous_Spending', hue='Purchase', data=df, palette={'Yes': ACCENT_GREEN, 'No': ACCENT_RED}, legend=False, ax=ax, width=0.45)
    ax.set_title('EDA: Previous Spending vs Purchase Decision', fontsize=12, pad=12, fontweight='bold')
    ax.set_xlabel('Purchase Decision', fontsize=11)
    ax.set_ylabel('Previous Spending (₹)', fontsize=11)
    apply_dark_theme(fig, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'previous_spending_vs_purchase.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    # 6. Discount Offered vs Purchase (Bar Chart)
    fig, ax = plt.subplots(figsize=(6, 4))
    avg_discount = df.groupby('Purchase')['Discount_Percent'].mean()
    bars = ax.bar(avg_discount.index, avg_discount.values, color=[ACCENT_RED, ACCENT_GREEN], width=0.45, edgecolor=TEXT_COLOR)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.6, f'{yval:.1f}% avg', 
                ha='center', va='bottom', color=TEXT_COLOR, fontweight='bold', fontsize=11)
    ax.set_title('EDA: Average Discount Offered vs Purchase', fontsize=12, pad=12, fontweight='bold')
    ax.set_xlabel('Purchase Decision', fontsize=11)
    ax.set_ylabel('Average Discount (%)', fontsize=11)
    ax.set_ylim(0, max(avg_discount.values) + 5)
    apply_dark_theme(fig, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'discount_vs_purchase.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    # 7. Confusion Matrix Plot
    if y_test is not None and y_pred is not None:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Pred: No (0)', 'Pred: Yes (1)'],
                    yticklabels=['Actual: No (0)', 'Actual: Yes (1)'],
                    annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
        ax.set_title('Step 6: Logistic Regression Confusion Matrix', fontsize=12, pad=12, fontweight='bold')
        apply_dark_theme(fig, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=150, facecolor=DARK_BG)
        plt.close()

def init_db(db_path='database/predictions.db'):
    """
    Initializes the SQLite database for prediction history.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER NOT NULL,
            previous_purchases INTEGER NOT NULL,
            website_time REAL NOT NULL,
            products_viewed INTEGER NOT NULL,
            previous_spending REAL NOT NULL,
            discount REAL NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def train_and_evaluate(csv_path='data/customer_purchase_data.csv', model_path='models/purchase_model.pkl'):
    """
    Trains the Logistic Regression model pipeline and evaluates it.
    """
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = generate_dataset(n_samples=200, random_seed=42)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    # Features and target
    feature_cols = ['Age', 'Previous_Purchases', 'Website_Time_Minutes', 'Products_Viewed', 'Previous_Spending', 'Discount_Percent']
    X = df[feature_cols]
    y = df['Purchase'].map({'No': 0, 'Yes': 1})

    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline with StandardScaler and LogisticRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    # Generate plots
    save_plots(df, y_test, y_pred)

    # Initialize Database
    init_db()

    print("=" * 60)
    print("CUSTOMER PURCHASE PREDICTION - MODEL TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Dataset Samples: {len(df)}")
    print(f"Training Samples (80%): {len(X_train)}")
    print(f"Testing Samples (20%):  {len(X_test)}")
    print(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"Target Distribution: Yes={sum(y==1)}, No={sum(y==0)}")
    print("-" * 60)
    print("TEST EVALUATION METRICS:")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    print("=" * 60)

    return {
        'df': df,
        'pipeline': pipeline,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'metrics': {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'confusion_matrix': cm.tolist()
        }
    }

if __name__ == '__main__':
    train_and_evaluate()
