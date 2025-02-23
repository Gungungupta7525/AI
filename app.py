from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
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

        # Read the uploaded CSV file
        df = pd.read_csv(filepath)

        # Get the first 5 rows (head)
        head_data = df.head().values.tolist()

        # Get number of rows and columns
        num_rows, num_columns = df.shape

        # Get basic statistics: count of null values, mean, std, etc.
        column_stats = df.describe(include='all').transpose()  # This includes numeric and object columns

        # Add the count of null values to the stats
        column_stats['null_count'] = df.isnull().sum()

        # Add the data types to the stats
        column_stats['data_type'] = df.dtypes

        # Convert the DataFrame stats to a dictionary for easy access in the template
        column_stats_dict = column_stats.to_dict(orient='index')

        # Pass all the values to the template
        return render_template('select_columns.html', 
                               columns=df.columns, 
                               head_data=head_data, 
                               num_rows=num_rows, 
                               num_columns=num_columns, 
                               column_stats=column_stats_dict)

    else:
        return "Invalid file type. Please upload a CSV file."


@app.route('/predict', methods=['POST'])
def predict():
    # Load the uploaded CSV file
    filename = request.form['filename']
    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Get selected features and target from the form
    features = request.form.getlist('features')
    target = request.form['target']

    # Check if selected features and target columns exist in the dataset
    if not all(col in data.columns for col in features):
        return "One or more selected features are not in the dataset."

    if target not in data.columns:
        return "The selected target column is not in the dataset."

    # Handle missing values based on user's selection
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

    # Handle duplicate rows if requested
    if 'remove_duplicates' in request.form:
        data.drop_duplicates(inplace=True)

    # Split the data into features and target
    X = data[features]
    y = data[target]

    # Handle categorical features with OneHotEncoder if necessary
    categorical_columns = [col for col in features if data[col].dtype == 'object']
    if categorical_columns:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        # Apply OneHotEncoding to categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_columns)
            ],
            remainder='passthrough'
        )
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('classifier', LogisticRegression(solver=request.form['solver'], max_iter=int(request.form['max_iter'])))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model_pipeline.fit(X_train, y_train)
        predictions = model_pipeline.predict(X_test)
    else:
        # If no categorical columns, use LogisticRegression directly
        model = LogisticRegression(solver=request.form['solver'], max_iter=int(request.form['max_iter']))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # Render results in a template
    return render_template('results.html', 
                           predictions=predictions.tolist(), 
                           accuracy=accuracy, 
                           precision=precision, 
                           recall=recall, 
                           f1_score=f1)


if __name__ == '__main__':
    app.run(debug=True)
