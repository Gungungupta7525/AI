from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Folder setup
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------ HOME ------------------
@app.route('/')
def index():
    return render_template('index.html')

# ------------------ UPLOAD ------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'datafile' not in request.files:
        return "No file part"

    file = request.files['datafile']

    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)

        # Dataset info
        head_data = df.head().values.tolist()
        num_rows, num_columns = df.shape

        column_stats = df.describe(include='all').transpose()
        column_stats['null_count'] = df.isnull().sum()
        column_stats['data_type'] = df.dtypes
        column_stats_dict = column_stats.to_dict(orient='index')

        # Only numeric columns for graph dropdown
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        return render_template('select_columns.html',
                               columns=df.columns,
                               numeric_columns=numeric_columns,
                               head_data=head_data,
                               num_rows=num_rows,
                               num_columns=num_columns,
                               column_stats=column_stats_dict,
                               filename=filename)

# ------------------ GRAPH PLOTTING ------------------
@app.route('/plot', methods=['POST'])
def plot():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    filename = request.form['filename']
    column = request.form['column']
    graph_type = request.form['graph_type']

    df = pd.read_csv(os.path.join('uploads', filename))

    plt.figure()

    if graph_type == 'hist':
        df[column].hist()
    elif graph_type == 'line':
        df[column].plot()
    elif graph_type == 'box':
        df.boxplot(column=column)

    plot_path = f"static/plot.png"
    plt.savefig(plot_path)
    plt.close()

    return render_template('select_columns.html',
                           columns=df.columns,
                           numeric_columns=df.select_dtypes(include=['number']).columns,
                           filename=filename,
                           plot_url=plot_path)

# ------------------ MODEL TRAINING ------------------
@app.route('/predict', methods=['POST'])
def predict():
    filename = request.form['filename']
    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    features = request.form.getlist('features')
    target = request.form['target']

    if not all(col in data.columns for col in features):
        return "Invalid feature selection"

    if target not in data.columns:
        return "Invalid target column"

    # Handle missing values
    option = request.form['missing_values']

    if option == 'drop':
        data.dropna(subset=features + [target], inplace=True)

    elif option == 'fill_mean':
        data[features] = data[features].fillna(data[features].mean())

    elif option == 'fill_median':
        data[features] = data[features].fillna(data[features].median())

    elif option == 'fill_mode':
        for col in features:
            data[col] = data[col].fillna(data[col].mode()[0])

    # Remove duplicates
    if 'remove_duplicates' in request.form:
        data.drop_duplicates(inplace=True)

    y = data[target]

    # -------- TEXT HANDLING (FIXED) --------
    from sklearn.feature_extraction.text import TfidfVectorizer

    if any(data[col].dtype == 'object' for col in features):
        text_data = data[features].astype(str).agg(" ".join, axis=1)
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(text_data)
    else:
        X = data[features]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    model = LogisticRegression(
        solver=request.form['solver'],
        max_iter=int(request.form['max_iter'])
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    return render_template('results.html',
                           predictions=predictions.tolist(),
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           f1_score=f1)

# ------------------ RUN ------------------
if __name__ == '__main__':
    app.run(debug=True)