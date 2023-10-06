import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('./Datasets/boston.csv')

# Features and target variable
X = data[['rm']]  # Using only 'rm' feature for now
y = data['medv']

# Splitting the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Training the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the target for the test set
y_pred = lr.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot the regression line on the test data
plt.figure(figsize=(12, 6))
sns.scatterplot(x=X_test['rm'], y=y_test, color='blue', label='Actual Values', alpha=0.6)
plt.plot(X_test['rm'], y_pred, color='red', label='Regression Line')
plt.title('Regression Line on Test Data')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Home Value ($1000s)')
plt.legend()
plt.show()