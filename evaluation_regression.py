import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

data = {"Deneyim_Yılı" : [5,7,3,3,2,7,3,10,6,4,8,1,1,9,1] ,
        "Maaş" : [600,900,550,500,400,950,540,1200,900,550,1100,460,400,1000,380]}

df = pd.DataFrame(data)

y = df[["Maaş"]]

X = df[["Deneyim_Yılı"]]


reg_model = LinearRegression().fit(X,y)

b = reg_model.intercept_[0]

w1 = reg_model.coef_[0][0]

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Maaş")
g.set_xlabel("Deneyim_Yılı")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


y_pred = reg_model.predict(X)
#MSE
mean_squared_error(y,y_pred)

#RMSE
np.sqrt(mean_squared_error(y,y_pred))
#MAE
mean_absolute_error(y,y_pred)



















