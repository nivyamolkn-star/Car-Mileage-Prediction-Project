#  Car Mileage Prediction Project
# Topics Covered: Correlation & Regression, Linear Algebra, Calculus

# Step 1️: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2️: Create a simple dataset manually
# We'll simulate car data with engine size, weight, horsepower, and mileage (mpg)
np.random.seed(42)

engine_size=np.random.uniform(1.0,5.0,50)    # in liters
weight=np.random.uniform(800,2000,50)        #in kilograms
horsepower=np.random.uniform(60,250,50)      #in HP
# Mileage is negatively correlated with engine size, weight, and horsepower
mileage=50-(engine_size*3.5)-(weight*0.005)-(horsepower*0.03)+np.random.normal(0,2,50)

# Step 3️: Create a DataFrame
df=pd.DataFrame({
    "Engine_size(l)":engine_size,
    "Weight(kg)":weight,
    "Horsepower":horsepower,
    "Mileage":mileage
})

# Step 4️: Display dataset
print("Sample Data:")
print(df.head())

# Step 5️: Correlation Analysis
print("\n Correlation Matrix:")
corr_matrix=df.corr()
print(corr_matrix)

# Visualization of correlation
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# Step 6️: Linear Regression (Mileage vs Engine Size)
from sklearn.linear_model import LinearRegression

x=df[["Engine_size(l)"]]      # independent variable
y=df[["Mileage"]]              # dependent variable

model=LinearRegression()
model.fit(x,y)


# Regression line prediction
y_pred=model.predict(x)

# Visualization
plt.scatter(x,y,color="blue",label="Actual Data")
plt.plot(x,y_pred,color="red",label=" Regression Line")
plt.title("Mileage vs Engine Size")
plt.xlabel("Engine Size(L)")
plt.ylabel("Mileage(km/L)")
plt.legend()
plt.show()

print(f"Regression Coeffiicient(Slope):{model.coef_[0][0]:.3f}")
print(f"Intercept:{model.intercept_[0]:.3f}")

# Step 7️: Linear Algebra View (Matrix Form)
# y = Xβ + ε
# Here, we’ll manually compute β = (XᵀX)⁻¹ Xᵀy

x_matrix=np.c_[np.ones(len(x)),x]
y_vector=y.values.reshape(-1,1)

# Compute coefficients using matrix operations
beta=np.linalg.inv(x_matrix.T @ x_matrix) @ (x_matrix.T @ y_vector)
print("\n Regression Coefficients using Linear Algebra:")
print(beta)


# Step 8️: Calculus View — Gradient Descent Optimization
# We'll minimize Mean Squared Error (MSE) manually using calculus

# Initialize parameters
m,c=0,0      # slope and intercept
L=0.001      # learning rate
epochs=1000

for i in range(epochs):
  y_pred=m*x.values.flatten()+c
  D_m=(-2/len(x))*sum(x.values.flatten()*(y.values.flatten()-y_pred))
  D_c=(-2/len(x))*sum(y.values.flatten()-y_pred)
  m=m-L*D_m
  c=c-L*D_c

print("\n Gradient Descent Results:")
print(f"Estimated Slope(m):{m:.3f}")
print(f"Estimated Intercept(c):{c:.3f}")


# Visualization of gradient descent regression line
plt.scatter(x,y,color="blue",label="Actual Data")
plt.plot(x,m*x.values+c,color="green",label="Gradient Descent Line")
plt.legend()
plt.title("Mileage Prediction using Gradient Descent")
plt.xlabel("Engine Size(L)")
plt.ylabel("Mileage(km/L)")
plt.show()


