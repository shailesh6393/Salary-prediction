import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your data
data = pd.read_csv(r"C:\Users\admin\Desktop\VAM PROJECT\employee_data.csv")

# One-Hot Encoding (if any categorical columns need to be encoded)
# Replace 'Course' with relevant columns in your dataset if necessary
# Example: data = pd.get_dummies(data, columns=["ColumnName"], drop_first=True)

# Check data info to verify relevant columns
print(data.info())

# Assuming 'Years_of_Experience' is your feature and 'Salary' is the target
X = data[['Years_Experience']]
y = data['Salary']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-Squared Value:", r2)

# Visualization: Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()

# Print model parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'data' with columns 'Years_Experience' and 'Salary'
plt.scatter(data['Years_Experience'], data['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()
