from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

app = Flask(__name__)

# Read and preprocess the data
file_name = 'LA.csv'  # Replace with your file name
df = pd.read_csv(file_name)
df = df.dropna()

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'].astype(str))
df['Married'] = le.fit_transform(df['Married'].astype(str))
df['Dependents'] = le.fit_transform(df['Dependents'].astype(str))
df['Education'] = le.fit_transform(df['Education'].astype(str))
df['Self_Employed'] = le.fit_transform(df['Self_Employed'].astype(str))
df['Credit_History'] = le.fit_transform(df['Credit_History'].astype(str))
df['Property_Area'] = le.fit_transform(df['Property_Area'].astype(str))
df['Loan_Status'] = le.fit_transform(df['Loan_Status'].astype(str))

# Calculate loan amount ratio and total income ratio for each row
df['LoanAmountRatio'] = 1 - df['LoanAmount'] / df['Total_Income']
df['TotalIncomeRatio'] = (df['Total_Income'] - df['LoanAmount']) / df['Total_Income']

X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'TotalIncomeRatio',
        'LoanAmountRatio', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']].values
y = df['Loan_Status'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/logistic_regression')
def logistic_regression():
    return render_template('logistic_regression.html')
@app.route('/logistic_regression_results')
def logistic_regression_results():
    return render_template('logistic_regression_results.html')

@app.route('/logistic_regression_ui', methods=['POST', 'GET'])
def logistic_regression_ui():
    if request.method == 'POST':
        gender = int(float(request.form['gender']))
        married = int(float(request.form['married']))
        dependents = int(float(request.form['dependents']))
        education = int(float(request.form['education']))
        self_employed = int(float(request.form['self_employed']))
        total_income = float(request.form['total_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = float(request.form['loan_term'])
        credit_history = int(float(request.form['credit_history']))
        property_area = int(float(request.form['property_area']))

        loan_amount_ratio = (1 - loan_amount / total_income) * 1.1
        total_income_ratio = ((total_income - loan_amount) / total_income) * 1.1

        input_data = np.array([[gender, married, dependents, education,
                                 self_employed, total_income_ratio, loan_amount_ratio, loan_term,
                                 credit_history, property_area]])

        input_data = scaler.transform(input_data)

        prediction_proba = clf.predict_proba(input_data)[0]
        formatted_proba = ["{:.5f}".format(prob) for prob in prediction_proba]
        prediction = "Loan Application Approved" if prediction_proba[1] >= 0.80 else "Loan Application Rejected"

        return render_template('logistic_regression_ui.html', gender=gender, married=married, dependents=dependents,
                               education=education, self_employed=self_employed, total_income=total_income,
                               loan_amount=loan_amount, loan_term=loan_term, credit_history=credit_history, property_area=property_area,
                               formatted_proba=formatted_proba, prediction=prediction)
    else:
        return render_template('logistic_regression_ui.html')
if __name__ == '__main__':
    # Use the PORT environment variable if available, otherwise default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

