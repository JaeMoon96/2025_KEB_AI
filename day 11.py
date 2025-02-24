# from sklearn.linear_model import LinearRegression
import numpy as np
# import jmlearn as jm
from jmlearn import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
#print(type(ls)) 출력결과 <class 'pandas.core.frame.DataFrame'>
#print(ls)
X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values
#print(x)

ls.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500,62_500,4, 9])
plt.show()

model = LinearRegression()
# model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

X_new = [[31721.3]]  # ROK 2020
print(model.predict(X_new))
# LinearRegression 5.90
# KNeighborsRegressor 5.70
