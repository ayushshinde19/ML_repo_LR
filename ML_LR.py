import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:\\Users\\admin\\Downloads\\diabetes (2) - diabetes (2).csv")
df

df.columns

df.info()

df.isna().sum()

df.head(10)

df.describe()

x = df[["Age"]]
y = df[['Outcome']]

df.shape

plt.scatter(df["Outcome"],df["Age"])
plt.xlabel("Age")
plt.ylabel("Outcome")
plt.title("garph Baes on Age and there outcome")

model = LinearRegression()

model.fit(x,y)

model.intercept_

model.coef_

yp = model.predict(x)

yp

from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.33 ,random_state=25)

LinearRegression().fit(x_train,y_train)

model.score(x_train,y_train)

yp = model.predict(x)

predictions  = model.predict(x_train)

plt.scatter(y_train,predictions)
plt.xlabel("Y TRAIN")
plt.ylabel("prediction Y")

from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

print("MAE:",metrics.mean_absolute_error(y_train, predictions))
print("MSE:",metrics.mean_squared_error(y_train, predictions))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_train, predictions)))

from sklearn.metrics import r2_score
r_square = r2_score(y,yp)
print("Coefficient of Determination", r_square)

#find the alary of the employee who have 12 15 years experience
import warnings
warnings.filterwarnings("ignore")
salary = np.array([12,15]).reshape(-1,1)
model.predict(salary)