import pandas as pd
import numpy as np
import joblib
import sklearn
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

rf_model = joblib.load('random_forest_model.pkl')
svc_model = joblib.load('svm_model.pkl')

def preprocess_data(input_data):
    StandardScaler = joblib.load('standard_scaler.pkl')

    numeric_columns = ['Sensor 1 (Temperature C)', 'Sensor 2 (Pressure KPa)',
       'Wind speed (m/s)', 'Vibration (Hz)', 'n (RPM)', 'Power Output (MW)',
       'Ambient Temperature (deg C)']

    input_data[numeric_columns] = StandardScaler.transform(input_data[numeric_columns])

    return input_data


st.title('Predict wind turbine failure events')


st.write('##### Choose a machine learning model for your prediction:')
model_choice = st.selectbox('Model', ('Random Forest', 'SVM'))

st.write('##### Upload your data as a CSV file:')
uploaded_file = st.file_uploader("Choose a csv file:", type="csv")
numeric_columns = ['Sensor 1 (Temperature C)', 'Sensor 2 (Pressure KPa)',
       'Wind speed (m/s)', 'Vibration (Hz)', 'n (RPM)', 'Power Output (MW)',
       'Ambient Temperature (deg C)']
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    missing_column = [col for col in numeric_columns if col not in data.columns]

    if missing_column:
        st.error(f"The uploaded dataset is missing the following required operational data: {', '.join(missing_column)}")
    else:
        original_data = data.copy()
    
        prepros_data = preprocess_data(data)

        label_encoder = joblib.load('label_encoder.pkl')
    
        if model_choice == 'Random Forest':
            failure_events = rf_model.predict(prepros_data)
            failure_events_label = label_encoder.inverse_transform(failure_events)
    
        else:
            failure_events = svc_model.predict(prepros_data)
            failure_events_label = label_encoder.inverse_transform(failure_events)
        
        if len(failure_events) == len(data):
            original_data['Failure Event code'] = failure_events
            original_data['Failure Event Label'] = failure_events_label
            st.success("Your prediction is ready, please click download!")
            failure_event_download = original_data.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=failure_event_download,
                file_name='failure_event_predictions.csv',
                mime='text/csv'
            )
        else:
            st.error("The number of predictions does not match the number of rows in the uploaded data.")
    
