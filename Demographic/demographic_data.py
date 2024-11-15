import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_excel('//Users/shreyavora/Desktop/COVID-Delirium/-Data-Science-Discovery-Covid-Delirium-Project-/demographics.xlsx', engine='openpyxl')

#data = pd.read_excel('demogpraphics.xlsx', engine='openpyxl')
data = data.iloc[:128]

print(data.columns)

selected_columns = ['Age at Admission', 'Sex at Birth', 'Race', 'Hispanic ethnicity', "SOFA", "WHO Scale", 'Delirium at any time during hospitalization']
data = data[selected_columns]

data['Age at Admission'] = pd.to_numeric(data['Age at Admission'], errors='coerce')
average_age = data['Age at Admission'].mean()
data['Age at Admission'].fillna(average_age, inplace=True)
data['Age at Admission'] = np.ceil(data['Age at Admission']).astype(int)
print(data.head())

print(data['Age at Admission'].isna().sum())
print(data['Sex at Birth'].isna().sum())
print(data['Hispanic ethnicity'].isna().sum())
print(data['SOFA'].isna().sum())
#there is one empty 
average_SOFA = data['SOFA'].mean()
data['SOFA'].fillna(average_SOFA, inplace=True)
data['SOFA'] = np.ceil(data['SOFA']).astype(int)
print(data.head())
print(data['WHO Scale'].isna().sum())
average_WHO = data['WHO Scale'].mean()
data['WHO Scale'].fillna(average_WHO, inplace=True)
data['WHO Scale'] = np.ceil(data['WHO Scale']).astype(int)
#print(data.head())
print(data['WHO Scale'].isna().sum())
data['Sex at Birth'] = data['Sex at Birth'].map({'Male': 0, 'Female': 1})
# Remove any leading or trailing spaces in the column (if any)
data['Hispanic ethnicity'] = data['Hispanic ethnicity'].str.strip()
# Check for any non-numeric or unexpected values in the column
print(data['Hispanic ethnicity'].unique())
# Convert the column to numeric, coercing non-numeric values to NaN
data['Hispanic ethnicity'] = pd.to_numeric(data['Hispanic ethnicity'], errors='coerce')
# If there are NaN values after conversion, fill them with 0
data['Hispanic ethnicity'].fillna(0, inplace=True)
print(data[['Age at Admission', 'Sex at Birth', 'Hispanic ethnicity']].dtypes)
print(data['Hispanic ethnicity'].unique())

from sklearn.preprocessing import OneHotEncoder

unique_values = data['Race'].unique()
print (len(unique_values))

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output = False).set_output(transform = "pandas")
ohetransform = ohe.fit_transform(data[['Race']])
ohetransform
data = pd.concat([data, ohetransform], axis = 1).drop(columns= ["Race"])
data.head()
print(data.head)
print(len(data.columns))
data.to_csv('Demo_data.csv', index=False, header=True, sep=',')

