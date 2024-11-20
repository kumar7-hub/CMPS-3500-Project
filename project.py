#############################################################
# Course: CMPS3500
# CLASS Project
# PYTHON IMPLEMENTATION: BASIC DATA ANALYSIS
# Date: 11/19/24
# Student 1: Snehal Kumar
# Student 2: Marley Zerr
# Student 3: 
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