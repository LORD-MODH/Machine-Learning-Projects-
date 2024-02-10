from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 

x=np.array([1,2,3,4,5,6]).reshape(-1,1) #independent variable
print(x)
y=np.array([2,4,5 ,5, 4, 5]) #dependent variable

#training and testing split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1)

#model selection
model=LinearRegression()

#fitting parameters
model.fit(x_train,y_train)

#predicting data
predict=model.predict(x_test)

# Plotting data points and the regression line

# Plotting training data in blue
plt.scatter(x_train, y_train, color='blue', label='Training data')

# Plotting testing data in green
plt.scatter(x_test, y_test, color='green', label='Testing data')

# Plotting the regression line
# We use the entire 'x' to show the line across the full range of x values.
# The line is calculated based on the model predictions across all 'x' values.
plt.plot(x, model.predict(x), color='red', label='Regression line')

# Adding legend to differentiate between training data, testing data, and the regression line
plt.legend()

# Show the plot
plt.show()
