import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns
#sns.set()
%matplotlib qt


#Data loading
data = pd.read_csv('real_estate_price_size.csv', header=0)

#data describing
data.describe()

#data plotting
plt.scatter(data['price'],data['size'])
plt.xlabel('Price')
plt.ylabel('Size')

#regression 
x = sm.add_constant(data['size'])
result = sm.OLS(data['price'], x).fit()
result.summary()

plt.scatter(data['price'],data['size'])
yhat = 223.1787*data['size'] + 1.019
fig = plt.plot(yhat, data['size'] , lw=4, c='orange')
plt.xlabel('Price')
plt.ylabel('Size')