from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
import io
import base64
from sklearn.ensemble import IsolationForest
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('visualizations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS uploads (
        id TEXT PRIMARY KEY,
        filename TEXT,
        insights TEXT,
        anomalies TEXT
    )''')
    conn.commit()
    conn.close()

# Generate visualizations
def generate_visualizations(df, col1, col2):
    img_files = {}
    
    # Bar Chart: Mean of numerical columns
    plt.figure(figsize=(10, 6))
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_cols) > 0:
        means = df[numerical_cols].mean()
        means.plot(kind='bar')
        plt.title('Mean Values of Numerical Columns')
        plt.ylabel('Mean')
        plt.xticks(rotation=45)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_files['bar'] = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
    
    # Scatter Plot: col1 vs col2
    if col1 in df.columns and col2 in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[col1], df[col2])
        plt.title(f'Scatter Plot: {col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_files['scatter'] = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
    
    # Heatmap: Correlation matrix
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numerical_cols].corr()
        sns.heatmap(corr, annot=True, cmap='viridis')
        plt.title('Correlation Heatmap')
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_files['heatmap'] = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
    
    return img_files

# AI Feature 1: Generate statistical insights
def generate_insights(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    insights = []
    for col in numerical_cols:
        stats = {
            'Column': col,
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Std': df[col].std()
        }
        insights.append(stats)
    return insights

# AI Feature 2: Anomaly detection
def detect_anomalies(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_cols) == 0:
        return "No numerical columns for anomaly detection."
    X = df[numerical_cols].dropna()
    if X.empty:
        return "No valid data for anomaly detection."
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    labels = iso_forest.fit_predict(X)
    anomalies = X[labels == -1].index.tolist()
    return f"Anomalies detected at rows: {anomalies}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded", columns=None)
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected", columns=None)
        if file:
            # Check file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            if file_size > MAX_FILE_SIZE:
                return render_template('index.html', error="File exceeds 100 MB", columns=None)
            file.seek(0)
            
            # Save file
            filename = str(uuid.uuid4()) + '.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read CSV
            try:
                df = pd.read_csv(filepath)
                columns = df.columns.tolist()
                return render_template('select_columns.html', columns=columns, filename=filename)
            except Exception as e:
                return render_template('index.html', error=f"Error reading CSV: {str(e)}", columns=None)
    
    return render_template('index.html', error=None, columns=None)

@app.route('/visualize', methods=['POST'])
def visualize():
    filename = request.form['filename']
    col1 = request.form['col1']
    col2 = request.form['col2']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        df = pd.read_csv(filepath)
        # Generate visualizations
        img_files = generate_visualizations(df, col1, col2)
        # Generate AI insights
        insights = generate_insights(df)
        anomalies = detect_anomalies(df)
        
        # Store in database
        conn = sqlite3.connect('visualizations.db')
        c = conn.cursor()
        c.execute('INSERT INTO uploads (id, filename, insights, anomalies) VALUES (?, ?, ?, ?)',
                  (filename, filename, str(insights), anomalies))
        conn.commit()
        conn.close()
        
        return render_template('visualize.html', img_files=img_files, insights=insights, anomalies=anomalies)
    except Exception as e:
        return render_template('index.html', error=f"Error processing data: {str(e)}", columns=None)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
