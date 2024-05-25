import pickle
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer

# Load pre-trained model and data processing components
model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Function to clean input data
def clean_input_data(input_dict):
    """
    Clean and prepare input data by converting strings to lower case,
    replacing spaces with underscores, and handling numeric conversions.
    """
    for key in input_dict:
        if isinstance(input_dict[key], str):
            input_dict[key] = input_dict[key].lower().replace(' ', '_')
    
    cleaned_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, str):
            try:
                if value.replace('.', '', 1).isdigit():
                    cleaned_dict[key] = float(value)
                else:
                    cleaned_dict[key] = value
            except ValueError:
                cleaned_dict[key] = value
        else:
            cleaned_dict[key] = value
    return cleaned_dict

# Function to validate and clean data
def validate_and_clean_data(df):
    """
    Validate and clean the data by removing or imputing invalid values.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Apply imputation to all numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int']).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
        df[col] = df[col].fillna('missing')  # Handling missing values for categorical data
          # Remove rows with excessive 'missing' values
    df = df.replace('missing', pd.NA)  # Convert 'missing' to NA for easier counting
    df = df.dropna(thresh=len(df.columns) * 0.5)  # Keep rows with at least 50% non-NA values
    df = df.drop_duplicates()  # Remove duplicate rows if any


    return df

def main():
    # Load images for display
    image = Image.open('images/icone.png')
    image2 = Image.open('images/image.png')
    
    st.image(image, use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))
    
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image2)
    st.title("Predicting Customer Churn")

    file_upload = None  # Initialize file_upload variable

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            customer_ids = data['customer_id']  # Save customer_id before processing
            data = data.drop(columns=['customer_id'])  # Drop the customer_id for processing
            data = validate_and_clean_data(data)
            cleaned_data = [clean_input_data(record) for record in data.to_dict(orient='records')]
            try:
                X = dv.transform(cleaned_data)
                y_pred = model.predict_proba(X)[:, 1]
                data['churn_probability'] = y_pred
                data['churn'] = y_pred >= 0.5
                data.insert(0, 'customer_id', customer_ids)
                output_file = 'predictions_with_churn.csv'
                data.to_csv(output_file, index=False)
                st.success(f"Predictions saved to {output_file}")
                churn_rate = data['churn'].mean()
                st.write(f"Churn rate: {churn_rate:.2f}")
                st.subheader("Processed Data")
                st.write(data)
            except ValueError as e:
                st.error(f"Invalid input in batch data: {e}")

    elif add_selectbox == 'Online':
        # Additional data processing code for the "Online" section
        gender = st.selectbox('Gender:', ['male', 'female'])
        seniorcitizen = st.selectbox('Customer is a senior citizen:', [0, 1])
        partner = st.selectbox('Customer has a partner:', ['yes', 'no'])
        dependents = st.selectbox('Customer has dependents:', ['yes', 'no'])
        phoneservice = st.selectbox('Customer has phoneservice:', ['yes', 'no'])
        multiplelines = st.selectbox('Customer has multiplelines:', ['yes', 'no', 'no_phone_service'])
        internetservice = st.selectbox('Customer has internetservice:', ['dsl', 'no', 'fiber_optic'])
        onlinesecurity = st.selectbox('Customer has onlinesecurity:', ['yes', 'no', 'no_internet_service'])
        onlinebackup = st.selectbox('Customer has onlinebackup:', ['yes', 'no', 'no_internet_service'])
        deviceprotection = st.selectbox('Customer has deviceprotection:', ['yes', 'no', 'no_internet_service'])
        techsupport = st.selectbox('Customer has techsupport:', ['yes', 'no', 'no_internet_service'])
        streamingtv = st.selectbox('Customer has streamingtv:', ['yes', 'no', 'no_internet_service'])
        streamingmovies = st.selectbox('Customer has streamingmovies:', ['yes', 'no', 'no_internet_service'])
        contract = st.selectbox('Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
        paperlessbilling = st.selectbox('Customer has a paperlessbilling:', ['yes', 'no'])
        paymentmethod = st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check', 'mailed_check'])
        tenure = st.number_input('Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
        monthlycharges = st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
        totalcharges = tenure * monthlycharges
        output = ""
        output_prob = ""

        # One-hot encode 'gender'
        gender_encoded = 1 if gender == 'female' else 0  # Assign 1 for female, 0 for male

        input_dict = {
            "gender": gender_encoded,
            "seniorcitizen": seniorcitizen,
            "partner": partner,
            "dependents": dependents,
            "phoneservice": phoneservice,
            "multiplelines": multiplelines,
            "internetservice": internetservice,
            "onlinesecurity": onlinesecurity,
            "onlinebackup": onlinebackup,
            "deviceprotection": deviceprotection,
            "techsupport": techsupport,
            "streamingtv": streamingtv,
            "streamingmovies": streamingmovies,
            "contract": contract,
            "paperlessbilling": paperlessbilling,
            "paymentmethod": paymentmethod,
            "tenure": tenure,
            "monthlycharges": monthlycharges,
            "totalcharges": totalcharges
        }

        if st.button("Predict"):
            input_dict = clean_input_data(input_dict)

            try:
                X = dv.transform([input_dict])
                y_pred = model.predict_proba(X)[0, 1]
                churn = y_pred >= 0.5
                output_prob = float(y_pred)
                output = bool(churn)
            except ValueError as e:
                st.error(f"Invalid input: {e}")
                return

        st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))
        pass


if __name__ == '__main__':
    main()
