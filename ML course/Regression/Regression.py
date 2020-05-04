import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm


#Linear Regression
data = pd.read_csv('Salary_Data.csv')

# Training and Test Set
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# Fitting to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Visualising Training Set
plt.scatter(X_train, y_train, c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')
plt.title('Salary vs Experience (Training Test)')
plt.xlabel('Years of Expirience')
plt.ylabel('Salary')
plt.show()


#Visualising Test Set
plt.scatter(X_test, y_test, c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')
plt.title('Salary vs Experience (Test Test)')
plt.xlabel('Years of Expirience')
plt.ylabel('Salary')
plt.show()



#======================================================================
#Multiple Linear Regression
data = pd.read_csv('50_Startups.csv')

# Training and Test Set
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values



#Categorical data
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy trap
X = X[:, 1:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)



X = np.append(np.ones((50, 1)).astype(int), X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

#======================================================================
#Polinomial Regression
data = pd.read_csv('Position_Salaries.csv')

# Training and Test Set
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# Fitting to the Training Set
#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Polinomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualising Linear
plt.scatter(X, y, c='red')
plt.plot(X, lin_reg.predict(X), c='blue')
plt.title('Truff or Bluff Experience')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

#Visualising Polynomial
plt.scatter(X, y, c='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), c='blue')
plt.title('Truff or Bluff Experience(Polynomial)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

#Predicting Linear
lin_reg.predict(6.5)

#Predicting Polynomial
lin_reg_2.predict(poly_reg.fit_transform(6.5))

#==========================================
#Support Vector Regression
data = pd.read_csv('Position_Salaries.csv')

# Training and Test Set
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# Feature Scaling because SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1, 1))

#SVR Regressor
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]).reshape(-1,1))))


#Visualising Linear
plt.scatter(X, y, c='red')
plt.plot(X, regressor.predict(X), c='blue')
plt.title('Truff or Bluff Experience(SVR)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

#==========================================
#Decision Trees Regression

data = pd.read_csv('Position_Salaries.csv')

# Training and Test Set
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

#Decision Trees Regressor
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(6.5)

#Visualising 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#==========================================
#Random Forest  Regression

data = pd.read_csv('Position_Salaries.csv')

# Training and Test Set
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict(6.5)

#Visualising Random Forest
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#=========================================
#Model Perfomance




