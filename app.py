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

@app.route('/')
def index():
    return render_template('index.html')

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

        # Head & stats
        head_data = df.head().values.tolist()
        num_rows, num_columns = df.shape
        column_stats = df.describe(include='all').transpose()
        column_stats['null_count'] = df.isnull().sum()
        column_stats['data_type'] = df.dtypes
        column_stats_dict = column_stats.to_dict(orient='index')

        # Plot graphs for numeric columns
              # Plot graphs for numeric columns
        plot_paths = []
        for column in df.select_dtypes(include=['number']).columns:
            # Histogram
            plt.figure()
            df[column].plot(kind='hist', title=f"{column} - Histogram")
            hist_file = f"{column}_hist.png"
            hist_path = os.path.join(STATIC_FOLDER, hist_file)
            plt.savefig(hist_path)
            plt.close()
            plot_paths.append(hist_file)

            # Line Plot
            plt.figure()
            df[column].plot(kind='line', title=f"{column} - Line Plot")
            line_file = f"{column}_line.png"
            line_path = os.path.join(STATIC_FOLDER, line_file)
            plt.savefig(line_path)
            plt.close()
            plot_paths.append(line_file)

            # Boxplot
            plt.figure()
            df.boxplot(column=column)
            plt.title(f"{column} - Boxplot")
            box_file = f"{column}_box.png"
            box_path = os.path.join(STATIC_FOLDER, box_file)
            plt.savefig(box_path)
            plt.close()
            plot_paths.append(box_file)


        return render_template('select_columns.html',
                               columns=df.columns,
                               head_data=head_data,
                               num_rows=num_rows,
                               num_columns=num_columns,
                               column_stats=column_stats_dict,
                               plot_paths=plot_paths)
    else:
        return "Invalid file type. Please upload a CSV file."

@app.route('/predict', methods=['POST'])
def predict():
    filename = request.form['filename']
    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    features = request.form.getlist('features')
    target = request.form['target']

    if not all(col in data.columns for col in features):
        return "One or more selected features are not in the dataset."
    if target not in data.columns:
        return "The selected target column is not in the dataset."

    missing_values_option = request.form['missing_values']
    if missing_values_option == 'drop':
        data.dropna(subset=features + [target], inplace=True)
    elif missing_values_option == 'fill_mean':
        data[features] = data[features].fillna(data[features].mean())
    elif missing_values_option == 'fill_median':
        data[features] = data[features].fillna(data[features].median())
    elif missing_values_option == 'fill_mode':
        for feature in features:
            data[feature] = data[feature].fillna(data[feature].mode()[0])

    if 'remove_duplicates' in request.form:
        data.drop_duplicates(inplace=True)

    X = data[features]
    y = data[target]

    categorical_columns = [col for col in features if data[col].dtype == 'object']
    if categorical_columns:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(), categorical_columns)],
            remainder='passthrough'
        )
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver=request.form['solver'], max_iter=int(request.form['max_iter'])))
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model_pipeline.fit(X_train, y_train)
        predictions = model_pipeline.predict(X_test)
    else:
        model = LogisticRegression(solver=request.form['solver'], max_iter=int(request.form['max_iter']))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

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

if __name__ == '__main__':
    app.run(debug=True)
