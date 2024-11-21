#############################################################
# Course: CMPS3500
# CLASS Project
# PYTHON IMPLEMENTATION: BASIC DATA ANALYSIS
# Date: 11/19/24
# Student 1: Snehal Kumar
# Student 2: Marley Zerr
# Student 3: Joseph Rivera
# Student 4: 
# Description: Implementation Basic Data Analysis Routines
#############################################################

# General Packages
import math
import os
from pathlib import Path

# data handling libraries
import pandas as pd
import numpy as np
# from tabulate import tabulate

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
from keras.layers import Dense

def summarize_numerical_column_with_deviation(data, num_col, group_col = 'Customer_ID', absolute_summary = True, median_standardization_summary = False):
    '''Summarize the numerical column and its median standardization based on customers using describe_numerical_column function.'''
    Summary_dict = {}
        
    if median_standardization_summary == True:
        default_MAD = return_max_MAD(data, num_col, group_col)
        num_col_standardization = data.groupby(group_col)[num_col].apply(median_standardization, default_value = default_MAD)
        # print(f'Median standardization for {num_col}:\n')
        # Summary_dict[f'Median standardization of {num_col}'] = describe_numerical_column(num_col_standardization, f'Median standardization of {num_col}')
        Summary_dict['Max. MAD'] = default_MAD

    return Summary_dict

def return_max_MAD(data, num_col, group_col = 'Customer_ID'):
    '''Return max value of median absolute devaition(MAD) from within the customers for num_col'''
    return (data.groupby(group_col)[num_col].agg(lambda x: (x - x.median()).abs().median())).max()

def median_standardization(x, default_value):
    '''Transform series or dataframe to its devaition from median with respect to Median absolute deviation(MAD) i.e. median standardization.'''
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
    
def return_mode(x):
    '''Return nan if no mode exists in given series or return minimum mode'''
    modes = x.mode()
    if len(modes) == 0:
        return np.nan
    return modes.min()
    
def forward_backward_fill(x):
    '''Perform forward fill then backward fill on given series or dataframe'''
    return x.fillna(method='ffill').fillna(method='bfill')

def return_mode_median_filled_int(x):
    '''Return back series by filling with mode(in case there is one mode) else fill with integer part of median'''
    modes = x.mode()
    if len(modes) == 1:
        return x.fillna(modes[0])
    else:
        return x.fillna(int(modes.median()))
    
def return_mode_average_filled(x):
    '''Return back series by filling with mode(in case there is one mode) else fill with average of modes'''
    modes = x.mode()
    if len(modes) == 1:
        return x.fillna(modes[0])
    else:
        return x.fillna(modes.mean())
    
def fill_month_history(x):
    '''Return months filled data for 8-months period'''
    first_non_null_idx = x.argmin()
    first_non_null_value = x.iloc[first_non_null_idx]
    return pd.Series(first_non_null_value + np.array(range(-first_non_null_idx, 8-first_non_null_idx)), index = x.index)


def loadData(dataset):
    return pd.read_csv(f"{dataset}")


def cleanData(df):
    # Drop 'Name' column
    df.drop(columns = ['Name'], inplace = True)
    # Replace underscores with empty string in 'Age' column 
    df['Age'] = df['Age'].str.replace('_', '')
    # Convert 'Age' column to integer
    df['Age'] = df['Age'].astype(int) 
    # Drop 'SSN' column
    df.drop(columns = ['SSN'], inplace = True)
    # Replace underscores with null in 'Occupation' column
    df['Occupation'][df['Occupation'] == '_______'] = np.nan
    # Replace underscores with empty string in 'Annual_Income' column
    df['Annual_Income'] = df['Annual_Income'].str.replace('_', '')
    # Convert 'Annual_Income' column to float
    df['Annual_Income'] = df['Annual_Income'].astype(float)
    # Replace underscores with empty string in 'Num_of_Loan' column and convert to integer
    df['Num_of_Loan'] = df['Num_of_Loan'].str.replace('_', '').astype(int)
    # Replace underscores with empty string in 'Num_of_Delayed_Payment' column and convert to float
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.replace('_', '').astype(float)
    # Replace underscores with null in 'Changed_Credit_Limit' column
    df['Changed_Credit_Limit'][df['Changed_Credit_Limit'] == '_'] = np.nan
    # Convert 'Changed_Credit_Limit' column to float
    df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].astype(float)
    # Replace underscores with null in 'Credit_Mix' column
    df['Credit_Mix'][df['Credit_Mix'] == '_'] = np.nan
    # Replace underscores with empty string in 'Outstanding_Debt' column
    df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace('_', '') 
    # Convert 'Outstanding_Debt' column to float
    df['Outstanding_Debt'] = df['Outstanding_Debt'].astype(float)
    # Replace underscores with empty string in 'Amount_invested_monthly' column and convert to float
    df['Amount_invested_monthly'] = df['Amount_invested_monthly'].str.replace('_', '').astype(float)
    # Replace weird values with null in 'Payment_Behaviour' column
    df['Payment_Behaviour'][df['Payment_Behaviour'] == '!@9#%8'] = np.nan # go back and check
    # Replace invalid values with null in 'Monthly_Balance' column
    df['Monthly_Balance'][df['Monthly_Balance'] == '__-333333333333333333333333333__'] = np.nan # go back and check
    # Convert 'Monthly_Balance' column to float
    df['Monthly_Balance'] = df['Monthly_Balance'].astype(float)
    # Drop 'ID' column
    df.drop(columns = 'ID', inplace = True)
    # Replace negative and high positive values above 100 to null in 'Age' column
    df['Age'][(df['Age'] > 100) | (df['Age'] <= 0)] = np.nan
    # Replace outliers with null in 'Age' column 
    df['Age'][df.groupby('Customer_ID')['Age'].transform(median_standardization, default_value = return_max_MAD(df, 'Age')) > 80] = np.nan
    # Fill missing values in 'Age' column
    df['Age'] =  df.groupby('Customer_ID')['Age'].transform(forward_backward_fill).astype(int)
    # Fill with same profession in 'Occupation' column
    df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(forward_backward_fill)
    # Choose minumum mode where two modes exist for same monthly inhand salary 
    df['Annual_Income'][df['Monthly_Inhand_Salary'].notnull()] = df[df['Monthly_Inhand_Salary'].notnull()].groupby(['Customer_ID', 'Monthly_Inhand_Salary'], group_keys = False)['Annual_Income'].transform(return_mode)
    # Replace null values with nearby monthly inhand salary of a same annual income value in 'Monthly_Inhand_Salary column'
    df['Monthly_Inhand_Salary'] = df.groupby(['Customer_ID', 'Annual_Income'], group_keys = False)['Monthly_Inhand_Salary'].transform(forward_backward_fill)
    df['Annual_Income'][df['Monthly_Inhand_Salary'].isnull()] = np.nan
    # Set 'Annual_Income' and 'Monthly_Inhand_Salary' column to null for particular customer
    df.loc[[34042], ['Annual_Income', 'Monthly_Inhand_Salary']] = np.nan # go back and check
    # Make data same 
    df['Annual_Income'] = df.groupby('Customer_ID')['Annual_Income'].transform(forward_backward_fill)
    df['Monthly_Inhand_Salary'] = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].transform(forward_backward_fill)
    # Replace negative values in 'Num_Bank_Accounts' column with null
    df['Num_Bank_Accounts'][df['Num_Bank_Accounts'] < 0] = np.nan
    # Replace large median standardization values with null in 'Num_Bank_Accounts' column
    df['Num_Bank_Accounts'][df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Bank_Accounts')).abs() > 2] = np.nan
    # Fill missing values and covert to integer in 'Num_Bank_Accounts' column
    df['Num_Bank_Accounts'] = df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(forward_backward_fill).astype(int)
    # Replace large median standardization values with null in 'Num_Credit_Card' column
    df['Num_Credit_Card'][df.groupby('Customer_ID')['Num_Credit_Card'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Credit_Card')).abs() > 2] = np.nan
    # Fill missing values and convert to integer in 'Num_Credit_Card' column
    df['Num_Credit_Card'] = df.groupby('Customer_ID')['Num_Credit_Card'].transform(forward_backward_fill).astype(int)
    # Fill customer's records with their median
    df['Interest_Rate'] = df.groupby('Customer_ID')['Interest_Rate'].transform(lambda x: x.median())
    # Fill 'Num_of_Loan' column with the number of loans
    num_of_loans = df['Type_of_Loan'].str.split(', ').str.len()
    df['Num_of_Loan'][num_of_loans.notnull()] = num_of_loans[num_of_loans.notnull()]
    # Set to 0 if no loans
    df['Num_of_Loan'][num_of_loans.isnull()] = 0
    # Fill missing values and convert to integer in 'Num_of_Loan' column
    df['Num_of_Loan'] = df.groupby('Customer_ID')['Num_of_Loan'].transform(forward_backward_fill).astype(int)
    # Replace null values with 'No Loan' in 'Type_of_Loan' column
    df['Type_of_Loan'].fillna('No Loan', inplace = True)
    temp_series = df['Type_of_Loan']
    # Number of loans
    temp_lengths = temp_series.str.split(', ').str.len().astype(int)
    temp_lengths_max = temp_lengths.max()
    for index, val in temp_lengths.items():
        temp_series[index] = (temp_lengths_max - val) * 'No Loan, ' + temp_series[index]
        
    temp = temp_series.str.split(pat = ', ', expand = True)
    # unique_loans = set()
    for col in temp.columns:
        temp[col] = temp[col].str.lstrip('and ')
        # unique_loans.update(temp[col].unique())
    temp.columns = [f'Last_Loan_{i}' for i in range(int(df['Num_of_Loan'].max()), 0, -1)]
    df = pd.merge(df, temp, left_index = True, right_index = True)
    # Drop 'Type_of_Loan' column
    df.drop(columns = 'Type_of_Loan', inplace = True)

    # No cleaning on Delay from due date

    # Replace Num_of_Delayed_Payment with NaN if value is an outlier (above threshold) or negative
    summary_num_delayed_payments = summarize_numerical_column_with_deviation(df, 'Num_of_Delayed_Payment', median_standardization_summary = True)
    df['Num_of_Delayed_Payment'][(df['Num_of_Delayed_Payment'] > summary_num_delayed_payments['Num_of_Delayed_Payment']['Outlier upper range']) | (df['Num_of_Delayed_Payment'] < 0)] = np.nan
    # Fill missing delayed payment counts using custom logic (mode with median fallback) for each customer, convert to integers
    df['Num_of_Delayed_Payment'] = df.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(return_mode_median_filled_int).astype(int)
    # Fill missing Changed_Credit_Limit values using custom logic (mode with average fallback) for each customer
    df['Changed_Credit_Limit'] = df.groupby('Customer_ID')['Changed_Credit_Limit'].transform(return_mode_average_filled)
    # Replace Num_Credit_Inquiries values with NaN if they are outliers (above upper threshold) or negative
    summary_num_credit_inquiries = summarize_numerical_column_with_deviation(df, 'Num_Credit_Inquiries', median_standardization_summary = True)
    df['Num_Credit_Inquiries'][(df['Num_Credit_Inquiries'] > summary_num_credit_inquiries['Num_Credit_Inquiries']['Outlier upper range']) | (df['Num_Credit_Inquiries'] < 0)] = np.nan
    # Fill missing credit inquiry counts using forward/backward fill within customer groups, convert to integers
    df['Num_Credit_Inquiries'] = df.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(forward_backward_fill).astype(int)
    # Fill missing Credit_Mix values using forward and backward fill within each customer group
    df['Credit_Mix'] = df.groupby('Customer_ID')['Credit_Mix'].transform(forward_backward_fill)

    # No nulls found in Outstanding debt so no cleaning
    # Credit Utilization ratio does not have any cleaning done to it

    # Extract years and months from 'Credit_History_Age' string (e.g., "5 Years and 3 Months") into separate numeric columns
    df[['Years', 'Months']] = df['Credit_History_Age'].str.extract('(?P<Years>\d+) Years and (?P<Months>\d+) Months').astype(float)
    # Convert separated years and months into total months of credit history
    df['Credit_History_Age'] = df['Years'] * 12 + df['Months']
    # Remove temporary Years and Months columns after combining them into Credit_History_Age
    df.drop(columns = ['Years', 'Months'], inplace = True)
    # Fill missing credit history durations using custom logic (fill_month_history) for each customer, convert to integers
    df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age'].transform(fill_month_history).astype(int)
    # Convert Payment_of_Min_Amount from categorical (Yes,No,NM) to numeric (1,0,NaN)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0, 'NM': np.nan})
    # Fill missing 'Payment_of_Min_Amount' values with most frequent value (mode) for each customer
    df['Payment_of_Min_Amount'] = df.groupby('Customer_ID')['Payment_of_Min_Amount'].transform(lambda x: x.fillna(x.mode()[0]))
    # Convert Payment_of_Min_Amount from binary (1,0) to categorical (Yes,No)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({1: 'Yes', 0: 'No'})
    # Replace EMI values with NaN where standardized deviation exceeds 10000 (extreme outliers)
    deviation_total_emi = df.groupby('Customer_ID')['Total_EMI_per_month'].transform(median_standardization, default_value = return_max_MAD(df, 'Total_EMI_per_month'))
    df['Total_EMI_per_month'][deviation_total_emi > 10000] = np.nan
    # Replace outlier EMI values (above upper threshold) with NaN
    summary_total_emi_per_month = summarize_numerical_column_with_deviation(df, 'Total_EMI_per_month', median_standardization_summary = True)
    df['Total_EMI_per_month'][(df['Total_EMI_per_month'] > summary_total_emi_per_month['Total_EMI_per_month']['Outlier upper range'])] = np.nan
    # Fill missing monthly investment amounts with the median value for each customer
    df['Amount_invested_monthly'] = df.groupby('Customer_ID')['Amount_invested_monthly'].transform(lambda x: x.fillna(x.median()))
    # Fill missing payment behaviors: use mode if customer has clear pattern, otherwise use forward/backward fill
    df['Payment_Behaviour'] = df.groupby('Customer_ID')['Payment_Behaviour'].transform(lambda x: return_mode(x) if len(x.mode()) == 1 else forward_backward_fill(x))
    #Fills in null values with the median monthly balance of the customers other properly entered monthly balances
    df['Monthly_Balance'] = df.groupby('Customer_ID')['Monthly_Balance'].transform(lambda x: x.fillna(x.median()))
    #Dropping month column
    df.drop(columns = ['Month'], inplace = True)
    #shuffle data
    df = df.sample(frac = 1) 
    #Rearranging the columns
    df = df.loc[:, ['Customer_ID', 'Age', 'Occupation', 'Annual_Income',
        'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
        'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Credit_History_Age',
        'Payment_of_Min_Amount', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance', 'Last_Loan_9', 'Last_Loan_8', 'Last_Loan_7',
        'Last_Loan_6', 'Last_Loan_5', 'Last_Loan_4', 'Last_Loan_3',
        'Last_Loan_2', 'Last_Loan_1',
        'Credit_Score']]

    return df





def main():
    option = 0
    df = {}

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
            print("Loading input data set:")
            print("***********************")

            current_time = pd.Timestamp.now()
            print(f"{current_time} Starting Script")
            
            # Loading training data set
            current_time = pd.Timestamp.now()
            print(f"{current_time} Loading training data set")
            df = loadData("credit_score_data.csv")
            loading_time = pd.Timestamp.now() - current_time

            # Displaying total columns read
            current_time = pd.Timestamp.now()
            print(f"{current_time} Total columns read: {len(df.columns)}")

            # Displaying total rows read
            current_time = pd.Timestamp.now()
            print(f"{current_time} Total rows read: {len(df)}\n")
            print(f"Time to load is: {loading_time}\n")

        elif option == 2:
            if len(df) == 0:
                print("No data loaded. Please load data first.\n")
                continue

            print("Processing input data set:")
            print("**************************")

            # Performing data clean up
            current_time = pd.Timestamp.now()
            print(f"{current_time} Performing Data Clean Up")
            df_cleaned = cleanData(df)
            cleaning_time = pd.Timestamp.now() - current_time

            # Displaying total rows after cleaning 
            current_time = pd.Timestamp.now()
            # print(f"{current_time} Total columns after cleaning is: {len(df_cleaned.columns)}")
            print(f"{current_time} Total rows after cleaning is: {len(df_cleaned)}\n")
            print(f"Time to process is: {cleaning_time}\n")

        elif option == 3:
            print("Printing Model details:")
            print("***********************")

        elif option == 4:
            print("Testing Model:")
            print("**************")


            











if __name__ == '__main__':
    main()