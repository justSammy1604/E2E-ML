from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
import polars as pl 

data = pl.read_csv("data/diabetes_complete_clean.csv")
X = data.drop("diabetes_binary") 
y = data["diabetes_binary"]

# print(data.is_unique().sum())

# print(data.n_unique())
# print(data.is_duplicated().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

scale = RobustScaler() 

X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

"""
"""
