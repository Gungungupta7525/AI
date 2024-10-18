from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')




@app.route('/upload', methods=['POST'])
def upload_file():
    if 'datafile' not in request.files:
        return "No file part"

    file = request.files['datafile']

    # Check if a file was selected
    if file.filename == '':
        return "No selected file"

    if file:  # If a file was uploaded
        # Save the file to a designated location (e.g., current directory)
        file.save('uploaded.csv')  # Save the uploaded file

        # Read the CSV file into a DataFrame
        df = pd.read_csv('uploaded.csv')

        # Render a template to select columns for prediction
        return render_template('select_columns.html', columns=df.columns)


@app.route('/predict', methods=['POST'])
def predict():
    # Load the uploaded CSV file
    data = pd.read_csv('uploaded.csv')

    # Get selected features and target from the form
    features = request.form.getlist('features')
    target = request.form['target']

    # Prepare the data
    X = data[features]
    y = data[target]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the logistic regression model
    model = LogisticRegression(solver=request.form['solver'], max_iter=int(request.form['max_iter']))
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Render results in a template
    return render_template('results.html', predictions=predictions.tolist(), accuracy=accuracy)


if __name__ == '__main__':
    app.run(debug=True)
