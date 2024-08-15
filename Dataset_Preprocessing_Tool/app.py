import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to display missing values


def display_missing_values(df):
    return df.isnull().sum()

# Function to handle missing values


def handle_missing_values(df):
    missing_before = display_missing_values(df)
    df.fillna(method='ffill', inplace=True)
    missing_after = display_missing_values(df)
    return df, missing_before, missing_after

# Function to encode categorical data


def encode_categorical_data(df):
    le_dict = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        le_dict[column] = le
    return df, le_dict

# Function to apply feature scaling


def feature_scaling(df):
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df


st.title('Dataset Preprocessing Tool')

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader('Original Dataset')
    st.write(df)

    # Show missing values in the original dataset
    st.subheader('Missing Values in Original Dataset')
    st.write(display_missing_values(df))

    # Options for operations
    if st.checkbox('Show Data Description'):
        st.subheader('Data Description')
        st.write(df.describe(include='all'))

    if st.checkbox('Handle Missing Values'):
        df_processed, missing_before, missing_after = handle_missing_values(df)
        st.subheader('Missing Values Before Handling')
        st.write(missing_before)
        st.subheader('Missing Values After Handling')
        st.write(missing_after)
        st.subheader('Dataset After Handling Missing Values')
        st.write(df_processed)
    else:
        df_processed = df

    if st.checkbox('Encode Categorical Data'):
        df_processed, le_dict = encode_categorical_data(df_processed)
        st.subheader('Dataset After Encoding Categorical Data')
        st.write(df_processed)

    if st.checkbox('Feature Scaling'):
        df_processed = feature_scaling(df_processed)
        st.subheader('Dataset After Feature Scaling')
        st.write(df_processed)

    # Download the processed dataset
    st.subheader('Download Processed Dataset')
    df_processed.to_csv('processed_data.csv', index=False)
    st.download_button(
        label="Download Processed Dataset",
        data=open('processed_data.csv', 'rb').read(),
        file_name='processed_data.csv',
        mime='text/csv'
    )
