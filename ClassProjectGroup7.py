#############################################################
# Course: CMPS3500
# CLASS Project
# PYTHON IMPLEMENTATION: BASIC DATA ANALYSIS
# Date: 11/19/24
# Student 1: Snehal Kumar
# Student 2: Marley Zerr
# Student 3: Joseph Rivera
# Student 4: Dominic Flores
# Description: Implementation Basic Data Analysis Routines
#############################################################

# General Packages
import math
import os
from pathlib import Path

# data handling libraries
import pandas as pd
from pandas import Timestamp
import numpy as np
from tabulate import tabulate

# visualization libraries
# from matplotlib import pyplot as plt
# import seaborn as sns

# extra libraries
import warnings
warnings.filterwarnings('ignore')

# Packages to support NN

# sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# tensorflow
import tensorflow as tf
from tensorflow import keras

# Keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input


# Return max value of median absolute devaition(MAD) from within the customers for num_col
def return_max_MAD(data, num_col, group_col = 'Customer_ID'):
    return (data.groupby(group_col)[num_col].agg(lambda x: (x - x.median()).abs().median())).max()

# Transform series or dataframe to its devaition from median with respect to Median absolute deviation(MAD) i.e. median standardization.
def median_standardization(x, default_value):
    med = x.median() 
    abs = (x - med).abs()
    MAD = abs.median()
    if MAD == 0:
        if ((abs == 0).sum() == abs.notnull().sum()): # When MAD is zero and all non-null values are constant in x
            return x * 0
        else:
            return (x - med)/default_value # When MAD is zero but all non-values are not same in x
    else:
        return (x - med)/MAD # When MAD is non-zero

# Return nan if no mode exists in given series or return minimum mode
def return_mode(x):
    modes = x.mode()
    if len(modes) == 0:
        return np.nan
    return modes.min()

# Perform forward fill then backward fill on given series or dataframe'
def forward_backward_fill(x):
    return x.fillna(method='ffill').fillna(method='bfill')

# Return back series by filling with mode(in case there is one mode) else fill with integer part of median
def return_mode_median_filled_int(x):
    modes = x.mode()
    if len(modes) == 1:
        return x.fillna(modes[0])
    else:
        return x.fillna(int(modes.median()))

# Return back series by filling with mode(in case there is one mode) else fill with average of modes  
def return_mode_average_filled(x):
    modes = x.mode()
    if len(modes) == 1:
        return x.fillna(modes[0])
    else:
        return x.fillna(modes.mean())
    
# Return months filled data for 8-months period
def fill_month_history(x):
    first_non_null_idx = x.argmin()
    first_non_null_value = x.iloc[first_non_null_idx]

    return pd.Series(first_non_null_value + np.array(range(-first_non_null_idx, 8-first_non_null_idx)), index = x.index)

# Calculates performance of multivariate classification model
def calculate_performance_multiclass(y_true, y_pred):
    # Args:
    #     y_true: The true labels.
    #     y_pred: The predicted labels.

    # Returns:
    #     A dictionary containing the calculated metrics.

    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision, Recall, and F1-score (macro-averaged)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')

    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics


def loadData(dataset):
    return pd.read_csv(dataset)


def processData(df):
    # Dropping unnecessary columns
    df.drop(columns = ['Unnamed: 0', 'Name', 'Month', 'SSN'], inplace = True)

    # Cleaning 'ID' column
    df['ID'] = df['ID'].astype('string')
    
    # Cleaning 'Age' column
    df['Age'] = df['Age'].str.replace('_', '').astype(int)
    df['Age'][(df['Age'] > 100) | (df['Age'] <= 0)] = np.nan
    df['Age'][df.groupby('Customer_ID')['Age'].transform(median_standardization, default_value=return_max_MAD(df, 'Age')) > 80] = np.nan
    df['Age'] = df.groupby('Customer_ID')['Age'].transform(forward_backward_fill).astype(int)
    
    # Cleaning 'Occupation' column
    df['Occupation'][df['Occupation'] == '_______'] = np.nan
    df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(forward_backward_fill).astype("string")
    
    # Cleaning 'Annual_Income' column
    df['Annual_Income'] = df['Annual_Income'].str.replace('_', '').astype(float)
    df['Annual_Income'][df['Monthly_Inhand_Salary'].notnull()] = df[df['Monthly_Inhand_Salary'].notnull()].groupby(['Customer_ID', 'Monthly_Inhand_Salary'], group_keys = False)['Annual_Income'].transform(return_mode)
    df['Annual_Income'][df['Monthly_Inhand_Salary'].isnull()] = np.nan
    Annual_Income_deviation = df.groupby('Customer_ID', group_keys = False)['Annual_Income'].apply(lambda x: (x - x.median())/x.median())
    indices = Annual_Income_deviation[Annual_Income_deviation > 500].index.tolist()
    df.loc[indices, ['Annual_Income', 'Monthly_Inhand_Salary']] = np.nan
    df['Annual_Income'] = df.groupby('Customer_ID')['Annual_Income'].transform(forward_backward_fill)
    
    # Cleaning 'Monthly_Inhand_Salary' column
    df['Monthly_Inhand_Salary'] = df.groupby(['Customer_ID', 'Annual_Income'], group_keys=False)['Monthly_Inhand_Salary'].transform(forward_backward_fill)
    df['Monthly_Inhand_Salary'][df['Annual_Income'].isnull()] = np.nan
    
    # Cleaning 'Num_of_Loan' column
    df['Num_of_Loan'] = df['Num_of_Loan'].str.replace('_', '').astype(int)
    num_of_loans = df['Type_of_Loan'].str.split(', ').str.len()
    df['Num_of_Loan'][num_of_loans.notnull()] = num_of_loans[num_of_loans.notnull()]
    df['Num_of_Loan'][num_of_loans.isnull()] = 0
    df['Num_of_Loan'] = df.groupby('Customer_ID')['Num_of_Loan'].transform(forward_backward_fill).astype(int)

    # Cleaning 'Type_of_Loan' column
    df['Type_of_Loan'].fillna('No Loan', inplace=True)

    temp_series = df['Type_of_Loan']
    temp_lengths = temp_series.str.split(', ').str.len().astype(int)
    temp_lengths_max = temp_lengths.max()
    for index, val in temp_lengths.items():
        temp_series[index] = (temp_lengths_max - val) * 'No Loan, ' + temp_series[index]
        
    temp = temp_series.str.split(pat = ', ', expand = True)
    for col in temp.columns:
        temp[col] = temp[col].str.lstrip('and ')

    temp.columns = [f'Last_Loan_{i}' for i in range(int(df['Num_of_Loan'].max()), 0, -1)]
    df = pd.merge(df, temp, left_index = True, right_index = True)
    df.drop(columns = 'Type_of_Loan', inplace = True)

    # Cleaning 'Last_Loan' columns
    for i in range(1, 10):
        df[f'Last_Loan_{i}'] = df[f'Last_Loan_{i}'].astype("string")
        df[f'Last_Loan_{i}'][df[f'Last_Loan_{i}'] == 'No Loan'] = np.nan
        df[f'Last_Loan_{i}'] = df.groupby('Customer_ID')[f'Last_Loan_{i}'].transform(forward_backward_fill).astype("string")
    
    # Cleaning 'Num_of_Delayed_Payment' column
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.replace('_', '').astype(float)
    df['Num_of_Delayed_Payment'] = df.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(return_mode_median_filled_int).astype(int)
    
    # Cleaning 'Changed_Credit_Limit' column
    df['Changed_Credit_Limit'][df['Changed_Credit_Limit'] == '_'] = np.nan
    df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].astype(float)
    df['Changed_Credit_Limit'] = df.groupby('Customer_ID')['Changed_Credit_Limit'].transform(return_mode_average_filled)
    
    # Cleaning 'Credit_Mix' column
    df['Credit_Mix'][df['Credit_Mix'] == '_'] = np.nan
    df['Credit_Mix'] = df.groupby('Customer_ID')['Credit_Mix'].transform(forward_backward_fill).astype("string")
    
    # Cleaning 'Outstanding_Debt' column
    df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace('_', '').astype(float)
    
    # Cleaning 'Amount_invested_monthly' column
    df['Amount_invested_monthly'] = df['Amount_invested_monthly'].str.replace('_', '').astype(float)
    df['Amount_invested_monthly'] = df.groupby('Customer_ID')['Amount_invested_monthly'].transform(lambda x: x.fillna(x.median()))
    
    # Cleaning 'Payment_Behaviour' column
    df['Payment_Behaviour'][~df['Payment_Behaviour'].str.match('^[A-Za-z_]+$')] = np.nan
    df['Payment_Behaviour'] = df.groupby('Customer_ID')['Payment_Behaviour'].transform(lambda x: return_mode(x) if len(x.mode()) == 1 else forward_backward_fill(x))
    df['Payment_Behaviour'] = df['Payment_Behaviour'].astype("string")
    
    # Cleaning 'Monthly_Balance' column
    df['Monthly_Balance'][~df['Monthly_Balance'].astype(str).str.match('^[-+]?(\d*\.)?\d+$')] = np.nan
    df['Monthly_Balance'] = df['Monthly_Balance'].astype(float)
    df['Monthly_Balance'] = df.groupby('Customer_ID')['Monthly_Balance'].transform(lambda x: x.fillna(x.median()))

    # Cleaning 'Num_Bank_Accounts' column
    df['Num_Bank_Accounts'][df['Num_Bank_Accounts'] < 0] = np.nan
    df['Num_Bank_Accounts'][df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Bank_Accounts')).abs() > 2] = np.nan
    df['Num_Bank_Accounts'] = df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(forward_backward_fill).astype(int)

    # Cleaning 'Num_Credit_Card' column
    df['Num_Credit_Card'][df.groupby('Customer_ID')['Num_Credit_Card'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Credit_Card')).abs() > 2] = np.nan
    df['Num_Credit_Card'] = df.groupby('Customer_ID')['Num_Credit_Card'].transform(forward_backward_fill).astype(int)
    
    # Cleaning 'Interest_Rate' column
    df['Interest_Rate'] = df.groupby('Customer_ID')['Interest_Rate'].transform(lambda x: x.median())

    # Cleaning 'Num_Credit_Inquiries' column
    df['Num_Credit_Inquiries'] = df.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(forward_backward_fill).astype(int)

    # Cleaning 'Credit_History_Age' column
    df[['Years', 'Months']] = df['Credit_History_Age'].str.extract('(?P<Years>\d+) Years and (?P<Months>\d+) Months').astype(float)
    df['Credit_History_Age'] = df['Years'] * 12 + df['Months']
    df.drop(columns=['Years', 'Months'], inplace=True)
    df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age'].transform(fill_month_history).astype(int)
    
    # Cleaning 'Payment_of_Min_Amount' column
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0, 'NM': np.nan})
    df['Payment_of_Min_Amount'] = df.groupby('Customer_ID')['Payment_of_Min_Amount'].transform(lambda x: x.fillna(x.mode()[0]))
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({1: 'Yes', 0: 'No'})
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].astype("string")
    
    # Cleaning 'Total_EMI_per_month' column
    deviation_total_emi = df.groupby('Customer_ID')['Total_EMI_per_month'].transform(median_standardization, default_value=return_max_MAD(df, 'Total_EMI_per_month'))
    df['Total_EMI_per_month'][deviation_total_emi > 10000] = np.nan

    # Cleaning 'Credit_Score' column
    df['Credit_Score'] = df['Credit_Score'].astype("string")
    
    # Shuffling and rearranging columns
    df = df.sample(frac=1) 
    df = df.loc[:, ['ID', 'Customer_ID', 'Age', 'Occupation', 'Annual_Income',
                    'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
                    'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                    'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
                    'Credit_Utilization_Ratio', 'Credit_History_Age',
                    'Payment_of_Min_Amount', 'Total_EMI_per_month',
                    'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance',
                    'Last_Loan_9', 'Last_Loan_8', 'Last_Loan_7', 'Last_Loan_6',
                    'Last_Loan_5', 'Last_Loan_4', 'Last_Loan_3', 'Last_Loan_2',
                    'Last_Loan_1', 'Credit_Score']]

    # df.to_csv("Credit_score_cleaned_data.csv", index = False)
    
    return df


def trainNeuralNetwork(df):
    # Dropping columns
    columns_to_drop_unrelated = ['Customer_ID']
    df.drop(columns=columns_to_drop_unrelated, inplace=True)

    continuous_features = [
        'Num_Credit_Card', 'Credit_Utilization_Ratio', 'Credit_History_Age',
        'Outstanding_Debt', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Delay_from_due_date'
    ]
    
    categorical_features = [
        'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour',
        'Last_Loan_1', 'Last_Loan_2', 'Last_Loan_3', 'Last_Loan_4', 'Last_Loan_5',
        'Last_Loan_6', 'Last_Loan_7', 'Last_Loan_8', 'Last_Loan_9'
    ]

    encoded_features = categorical_features

    target = ['Credit_Score']

    # Fill NA values before encoding
    for feature in encoded_features:
        df[feature] = df[feature].fillna('Unknown')

    # Scaling continuous features
    scaler = MinMaxScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    # Encoder for input features and target
    encoder = OneHotEncoder(handle_unknown='ignore')
    le = LabelEncoder()

    # Encoding categorical features and converting to DataFrame
    encoded_categorical = encoder.fit_transform(df[encoded_features])
    encoded_categorical_df = pd.DataFrame(encoded_categorical.toarray(), columns=encoder.get_feature_names_out(encoded_features))    
    df = pd.concat([df, encoded_categorical_df], axis=1)

    # Encoding target and converting to DataFrame
    encoded_target = encoder.fit_transform(df[target])
    encoded_target_df = pd.DataFrame(encoded_target.toarray(), columns=encoder.get_feature_names_out(target))
    df = pd.concat([df, encoded_target_df], axis=1)

    # print(df.info())

    features_for_model = list(encoded_categorical_df.columns)

    # for feature in features_for_model:
    #     print(feature)
    # print("\n")

    target_features = ['Credit_Score_Good', 'Credit_Score_Poor', 'Credit_Score_Standard']   

    # Defining input features and target
    X = df[features_for_model].values
    y = df[target_features].values

    # Splitting data into training and testing sets
    indices = np.arange(len(df))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.20, random_state=42)

    # Defining the model
    model = keras.Sequential()

    # Input layer
    model.add(Input(shape=(X_train.shape[1],)))

    # Hidden layers 
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))

    # Output layer
    model.add(keras.layers.Dense(3, activation="softmax"))

    # Compile the model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs = 50, batch_size = 25)

    # Make Predictions
    predictions = model.predict(X_test)

    # Convert encoded target and predictions back to original values
    y_tested = encoder.inverse_transform(y_test)
    y_predicted = encoder.inverse_transform(predictions) 

    # Calculating performance of model
    metrics = calculate_performance_multiclass(y_tested, y_predicted)
    print(f"\n[{Timestamp.now().strftime('%H:%M:%S')}] Model Accuracy: {metrics['accuracy']}")
    print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Model Precision: {metrics['precision']}")
    print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Model Recall: {metrics['recall']}")
    print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Model f1_score: {metrics['f1_score']}")
    print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Model Confusion Matrix: {metrics['confusion_matrix']}\n")


    return X_train, X_test, y_tested, y_predicted, test_indices
    

def generatePredictions(df, X_train, X_test, y_tested, y_predicted, test_indices):
    print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Generating prediction using selected Neural Network")
    print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Size of training set: {len(X_train)}")
    print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Size of testing set: {len(X_test)}")

    # Save predictions to csv file containing columns 'ID' and 'Credit_Score'
    test_ids = df['ID'].iloc[test_indices]
    predictions_df = pd.DataFrame({'ID': test_ids, 'Credit_Score': y_predicted.ravel()})
    predictions_df.to_csv('predictions.csv', index=False)

    print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Predictions generated (predictions.csv have been generated)....\n")



def main():
    df = {}
    X_train, X_test, y_tested, y_predicted, test_indices = None, None, None, None, None
    process_data = False
    trainNN = False 
    option = 0

    while option != 5:
        # Display the menu
        print("(1) Load data")
        print("(2) Process data")
        print("(3) Model details")
        print("(4) Test model")
        print("(5) Quit")

        option = int(input("\nSelect an option: "))
        print()

        if option == 1:
            file_option = 0
            load_file = None

            directory_path = os.getcwd()
            files = os.listdir(directory_path)
            csv_files = [file for file in files if file.endswith('.csv')]

            if len(csv_files) == 0:
                print("No CSV files found in current directory. Please add a CSV file to the directory.\n")
                continue

            # sort files alphabetically
            csv_files.sort()

            # print files in current directory
            print("Available CSV files:")
            print("********************")

            for i, file in enumerate(csv_files):
                print(f"({i+1}) {file}")

            file_option = int(input("\nSelect a CSV file to load: "))
            load_file = csv_files[file_option - 1]

            print("\nLoading input data set:")
            print("***********************")

            print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Starting Script")
            print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Loading training data set")

            # Loading training data set
            start_time = Timestamp.now()
            df = loadData(load_file)
            loading_time = (Timestamp.now() - start_time).total_seconds()

            # Displaying total columns read
            print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Total columns read: {len(df.columns)}")

            # Displaying total rows read
            print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Total rows read: {len(df)}\n")
            print(f"Time to load is: {loading_time:.2f} seconds\n")

        elif option == 2:
            if len(df) == 0:
                print("No data loaded. Please load data first.\n")
                continue

            print("Process (Clean) data:")
            print("*********************")

            print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Performing Data Clean Up")

            # Performing data clean up
            start_time = Timestamp.now()
            df = processData(df)
            cleaning_time = (Timestamp.now() - start_time).total_seconds()
            process_data = True

            # Displaying total rows after cleaning 
            # print(f"{Timestamp.now().strftime('%H:%M:%S')} Total columns after cleaning is: {len(df.columns)}")
            print(f"[{Timestamp.now().strftime('%H:%M:%S')}] Total rows after cleaning is: {len(df)}\n")
            print(f"Time to process is: {cleaning_time:.2f} seconds\n")

        elif option == 3:
            if len(df) == 0:
                print("No data loaded. Please load data first.\n")
                continue
            elif not process_data:
                print("Data not processed. Please process data first.\n")
                continue

            print("Train NN:")
            print("********")

            X_train, X_test, y_tested, y_predicted, test_indices = trainNeuralNetwork(df)
            trainNN = True

        elif option == 4:
            if len(df) == 0:
                print("No data loaded. Please load data first.\n")
                continue
            elif not process_data:
                print("Data not processed. Please process data first.\n")
                continue
            elif not trainNN:
                print("Nueral Network not trained. Please train NN first.\n")
                continue

            print("Generate Predictions:")
            print("********************")

            # Generating predictions
            generatePredictions(df, X_train, X_test, y_tested, y_predicted, test_indices)

            df = {}
            X_train, X_test, y_tested, y_predicted, test_indices = None, None, None, None, None
            process_data = False
            trainNN = False
        
        else:
            if option != 5:
                print("Invalid option. Please try again.\n")

    print("Exiting program....\n")


            


if __name__ == '__main__':
    main()