import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report,roc_curve


# After fitting your model

from sklearn.model_selection import train_test_split, cross_validate

from knn import y_prob
from linear_regression import X_train, X_test

df = pd.DataFrame({"Gerçek Değer" : [1,1,1,1,1,1,0,0,0,0],
                   "Model Olasılık Tahmini" : [1,1,1,1,0,1,1,0,0,0]})


# Özellikler (X) ve hedef değişken (y)
X = df[["Model Olasılık Tahmini"]]
y = df["Gerçek Değer"]  # DataFrame yerine Series olarak alınmalı

# Modeli eğit
log_model = LogisticRegression().fit(X, y)

# Sabit terim (intercept)
print("Intercept (Sabit):", log_model.intercept_)

# Katsayılar (coefficient)
print("Katsayılar (Coef):", log_model.coef_)

y_pred = log_model.predict(X)


print(classification_report(y,y_pred))
#precision :  0.83
# accuracy : 0.80
#recall : 0.83
# f1 score :  0.83

##Görev 2:
#accuracy = 905 / 1000 (0.905)
#Precision = 5 / 95 (0.0526)
#Recall = 5 / 10 (0.5)
#F1_Score = 2 * Precision * Recall / (Precision + Recall) (0.0952)

#Model genel olarak iyi bir doğruluk oranına sahip gibi görünüyor.
#Ama, Precision ve Recall çok düşük olduğu için sınıf dengesizliği olabilir.












