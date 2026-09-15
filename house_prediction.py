import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
df = pd.read_csv("housing.csv") # Accessing the csv file and write it down
                                # df is our entire table 
print(df)
print(df.head()) #First few rows
print(df.shape) #no. of rows and columns
print(df.info()) #col name + data types
print(df.describe()) #statistical summary

print(df.isnull().sum()) #Tells whether datasets has missing values

x = df[["area", "bedrooms","bathrooms","stories", "parking"]] #From our DataFrame df, take this 5 columns and store them in x
y = df["price"]      #Take only the price column and store it in y

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Display results
print("X_train:")
print(X_train)

print("X_test:")
print(X_test)

print("y_train:")
print(y_train)

print("y_test:")
print(y_test)


# Display shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Create model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Display what the model learned
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

y_pred = model.predict(X_test)

print("Predicted prices:")
print(y_pred)

print("Actual prices:")
print(y_test.values)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("R2 Score:", r2)

new_house = pd.DataFrame({
    "area": [2000],
    "bedrooms": [3],
    "bathrooms": [2],
    "stories": [2],
    "parking": [1]
})

prediction = model.predict(new_house)

print("Predicted house price:", prediction[0])