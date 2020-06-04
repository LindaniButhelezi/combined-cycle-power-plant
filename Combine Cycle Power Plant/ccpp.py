"""
Predicting the electrical output of a combined cycle power plant using multivariable linear regression
by L Buthelezi- BScEng(Chemical Engineering)
Location: Durban, South Africa
Email: L.Buthelezi@alumni.uct.ac.za
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
lst=plt.style.available
plt.style.use(lst[10])
plantData=pd.read_csv('data1.csv')
"""
This data has multiple independant variables (X values) which are:
    * ambient T, Vacuum, Ambient P and  
These are in column 0 to column 3 ===> iloc[:,:-1]

The dependant variable (Y) is the profit which is in column 5 (4 if zero indexing)===> iloc[:,4]
"""
X=plantData.iloc[:,:-1].values
X2=plantData.iloc[:,:-1].values
Y=plantData.iloc[:,4].values
print("Power Plant Data:\n",plantData.head(), "\n\n\n") # print the first 5 rows of data


sns.heatmap(plantData.corr()) #heat map showing interdependancies between variables
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

y_predict=regressor.predict(X_test)

 
print("Model Results:\n")
print("Ambient_T\tVacuum\t\tAmbient_P\tRH (%)\t\tPredicted MW\tActual MW")
for i in range(50):
    print("%0.2f\t\t%0.2f\t\t%0.2f\t\t%s\t\t%0.2f\t\t"%
    (X_test[i][0],X_test[i][1],X_test[i][2],X_test[i][3], y_predict[i]), round(y_test[i],2))

coeff=regressor.coef_
C=regressor.intercept_  

print("\n\nThe coefficients of X are: ", coeff)
print("The intercept is: ",C )

from sklearn.metrics import r2_score
R2=r2_score(y_test, y_predict)
print("The r2_value is, ",R2)

plt.scatter(y_test, y_predict, marker='x', label='Predicted by Model')
plt.plot(y_test, y_test, color="red", label='Target Profile')
plt.xlabel('Target Output (MW)')
plt.ylabel('Predicted Output (MW)')
plt.legend()
plt.show()

e=y_predict-y_test

fig2=plt.hist(e,15,edgecolor='black', linewidth=0.5, color='red', alpha=0.6)
plt.xlabel('diff. between Pridicted and Actual Output (MW)')
plt.ylabel('Number of Instances')
plt.show()

plt.scatter(X_test[:,0], y_predict,color='green', marker='x', label='Model Prediction')
plt.scatter(X_test[:,0], y_test, color="red", marker='.', label='Plant Data', alpha=0.5)
plt.xlabel('Ambient Temperature (deg.C)')
plt.ylabel('Predicted Output (MW)')

plt.legend()
plt.show()