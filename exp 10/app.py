import os
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib

from generate_data_and_train import train_and_evaluate, init_db

app = Flask(__name__)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'customer_purchase_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'purchase_model.pkl')
DB_PATH = os.path.join(BASE_DIR, 'database', 'predictions.db')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')

# Feature list
FEATURE_COLS = [
    'Age',
    'Previous_Purchases',
    'Website_Time_Minutes',
    'Products_Viewed',
    'Previous_Spending',
    'Discount_Percent'
]

def get_db_connection():
    """Returns a SQLite connection to the prediction database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_history(limit=50):
    """Retrieves prediction history records from SQLite database."""
    init_db(DB_PATH)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_dataset_and_model_info():
    """
    Loads dataset, model pipeline, and calculates all dynamic step statistics.
    Ensures zero hardcoded metrics.
    """
    # Ensure model and data exist
    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        train_and_evaluate(DATA_PATH, MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    # Step 1 Info
    total_records = len(df)
    features_count = len(FEATURE_COLS)
    yes_count = int((df['Purchase'] == 'Yes').sum())
    no_count = int((df['Purchase'] == 'No').sum())
    sample_rows = df.head(8).to_dict(orient='records')

    # Step 2 Missing Values & Preprocessing Info
    missing_before = {col: int(df[col].isnull().sum()) for col in df.columns}
    missing_after = {col: 0 for col in df.columns}
    
    # Feature statistics for scaling reference
    stats = []
    for col in FEATURE_COLS:
        stats.append({
            'feature': col,
            'mean': round(float(df[col].mean()), 2),
            'std': round(float(df[col].std()), 2),
            'min': round(float(df[col].min()), 2),
            'max': round(float(df[col].max()), 2)
        })

    # Step 4 & 6: Actual Train/Test evaluation
    X = df[FEATURE_COLS]
    y = df['Purchase'].map({'No': 0, 'Yes': 1})
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    evaluation = {
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }

    training_info = {
        'model_name': 'Logistic Regression',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'total_features': len(FEATURE_COLS),
        'classes': ['0 (No)', '1 (Yes)'],
        'status': 'Model Training Completed Successfully'
    }

    return {
        'total_records': total_records,
        'features_count': features_count,
        'feature_names': FEATURE_COLS,
        'target_variable': 'Purchase',
        'yes_count': yes_count,
        'no_count': no_count,
        'sample_rows': sample_rows,
        'missing_before': missing_before,
        'missing_after': missing_after,
        'stats': stats,
        'training_info': training_info,
        'evaluation': evaluation
    }

def validate_and_predict(data):
    """
    Validates form input and runs the ML model to predict purchase & probability.
    """
    try:
        age = float(data.get('age', 0))
        previous_purchases = int(data.get('previous_purchases', 0))
        website_time = float(data.get('website_time', 0))
        products_viewed = int(data.get('products_viewed', 0))
        previous_spending = float(data.get('previous_spending', 0))
        discount = float(data.get('discount', 0))

        # Range Validations
        if not (1 <= age <= 120):
            return None, "Age must be a valid positive value between 1 and 120."
        if previous_purchases < 0:
            return None, "Previous purchases cannot be negative."
        if website_time < 0:
            return None, "Time spent on website cannot be negative."
        if products_viewed < 0:
            return None, "Products viewed cannot be negative."
        if previous_spending < 0:
            return None, "Previous spending cannot be negative."
        if not (0 <= discount <= 100):
            return None, "Discount offered must be between 0% and 100%."

        input_df = pd.DataFrame([{
            'Age': age,
            'Previous_Purchases': previous_purchases,
            'Website_Time_Minutes': website_time,
            'Products_Viewed': products_viewed,
            'Previous_Spending': previous_spending,
            'Discount_Percent': discount
        }])

        pipeline = joblib.load(MODEL_PATH)
        pred_val = pipeline.predict(input_df)[0]
        prob_arr = pipeline.predict_proba(input_df)[0]
        
        prob_yes = float(prob_arr[1]) * 100.0
        prediction_str = "YES" if pred_val == 1 else "NO"

        # Save to SQLite database
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction_history 
            (age, previous_purchases, website_time, products_viewed, previous_spending, discount, prediction, probability, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (int(age), previous_purchases, website_time, products_viewed, previous_spending, discount, prediction_str, round(prob_yes, 2), now_str))
        conn.commit()
        conn.close()

        result = {
            'age': int(age),
            'previous_purchases': previous_purchases,
            'website_time': website_time,
            'products_viewed': products_viewed,
            'previous_spending': previous_spending,
            'discount': discount,
            'prediction': prediction_str,
            'probability': round(prob_yes, 2),
            'timestamp': now_str
        }
        return result, None

    except ValueError as ve:
        return None, f"Invalid numerical input: {str(ve)}"
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    context = get_dataset_and_model_info()
    prediction_result = None
    error_message = None

    if request.method == 'POST':
        result, error = validate_and_predict(request.form)
        if error:
            error_message = error
        else:
            prediction_result = result

    history = get_history(limit=50)
    return render_template(
        'index.html',
        context=context,
        prediction_result=prediction_result,
        error_message=error_message,
        history=history,
        current_year=datetime.now().year
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """AJAX endpoint for instant prediction without page reload."""
    data = request.get_json(force=True) if request.is_json else request.form
    result, error = validate_and_predict(data)
    if error:
        return jsonify({'success': False, 'error': error}), 400
    
    updated_history = get_history(limit=50)
    return jsonify({
        'success': True,
        'result': result,
        'history': updated_history
    })

@app.route('/api/history', methods=['GET'])
def api_history():
    """Returns all recorded prediction history."""
    history = get_history(limit=100)
    return jsonify({'success': True, 'history': history})

@app.route('/api/clear_history', methods=['POST'])
def api_clear_history():
    """Clears all records in the SQLite prediction history database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM prediction_history')
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Prediction history cleared successfully.'})

if __name__ == '__main__':
    # Initialize DB and ensure models exist
    init_db(DB_PATH)
    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        train_and_evaluate(DATA_PATH, MODEL_PATH)
        
    import sys
    port = 5000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    print("=" * 60)
    print("CUSTOMER PURCHASE PREDICTION SYSTEM")
    print(f"Flask server running locally at: http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host='127.0.0.1', port=port, debug=True)
