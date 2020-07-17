import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# reading data from  link

URL = "http://bit.ly/w-data"
data = pd.read_csv(URL)
print(data.head(5))


# simply plotting the scores as per hours to understand the data

data.plot(x='Hours', y='Scores', style='ok')
plt.title("Hours of study Vs Percentage")
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.show()


# to prepare data

X = data['Hours'].values.reshape(-1, 1)
y = data['Scores'].values.reshape(-1, 1)


# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# to train the model

regression = LinearRegression()
regression.fit(X_train, y_train)


# retrieve intercept and slope(y = mx + c)

print(regression.intercept_)
print(regression.coef_)


# predicting score from testing data (hours)

y_pred = regression.predict(X_test)


# comparing actual vs predicted

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)


# to visualize comparison

df1 = df.head(25)
df1.plot(kind='bar', figsize=(12, 8))
plt.title('Comparison')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# plot line with test data

plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# test for your study time(hours)  eg-9.25

time = np.array(float(input('Enter Study Time: ')))
predicted_score = regression.predict(time.reshape(1, -1))
print("Time Studied(hours)=", time)
print("Score predicted =", predicted_score)


# Model Evaluation
print("Mean Absolute Error :", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error :", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error :", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
