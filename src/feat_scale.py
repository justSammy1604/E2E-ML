from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
import polars as pl 

data = pl.read_csv("data/diabetes_cleaned.csv")
X = data.drop("diabetes_binary") 
y = data["diabetes_binary"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

scale = RobustScaler() 

X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)


""" positive_rows = data.filter(pl.col("diabetes_binary") == 1)
print(f"Total rows where diabetes_binary == 1: {positive_rows.height}")

# Display all rows. First, show the Polars DataFrame (may truncate if very wide):
print(positive_rows)

# For an untruncated full textual dump, convert to pandas and print all rows/columns.
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print(positive_rows.to_pandas().to_string(index=False)) """
"""
"""
# print(data.columns)