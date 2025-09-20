from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
import polars as pl

data = pl.read_csv("data/diabetes_complete_clean.csv")
X = data.drop("diabetes_binary")
y = data["diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

scale = RobustScaler()

X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

""" print(data.is_duplicated().sum())  # Check for duplicates
print(data.null_count().sum())  # Check for missing values
print(data['diabetes_binary'].value_counts())  # Check class distribution
print(data["diabetes_binary"].value_counts(normalize=True))  # Check class distribution """
