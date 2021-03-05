
import pandas as pd
url = 'https://drive.google.com/uc?export=download&id=1QrQ5Qqr-w4Qx-WPToDHrvFxM9iyS7O1N'
df = pd.read_csv(url)

df.head(10)

print('Average Quality: ' + str(df['quality'].mean())) 
print('maximum level of residual sugar: ' + str(df['residual sugar'].mean())) 
print('minimum level of residual sugar: ' + str(df['residual sugar'].mean()))

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

X = pd.DataFrame(df['volatile acidity'])
y = pd.DataFrame(df['quality'])
model = LinearRegression()
scores = []
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for i, (train, test) in enumerate(kfold.split(X, y)):
 model.fit(X.iloc[train,:], y.iloc[train,:])
 score = model.score(X.iloc[test,:], y.iloc[test,:])
 scores.append(score)
#print(scores)

plt.scatter(df['volatile acidity'], df['quality'])

#nlen = df['volatile acidity'].count
#X = df['volatile acidity'].values.reshape(nlen,1)
#y = pd.DataFrame(df['quality'])
#reg = LinearRegression().fit(X.iloc[train,:], Y.iloc[train,:])

#Ytar = pd.DataFrame(df['quality'])
Xsrc = pd.DataFrame(df.drop(['quality'],1))

model1 = LinearRegression()
scores = []
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for i, (train, test) in enumerate(kfold.split(X, y)):
 model1.fit(Xsrc.iloc[train,:], y.iloc[train,:])
 score = model1.score(Xsrc.iloc[test,:],y.iloc[test,:])
 scores.append(score)
#print(scores)

from sklearn import metrics
y_predall = model1.predict(Xsrc)
y_pred = model.predict(X)
print('Mean Squared error of one column: ' + str(metrics.mean_squared_error(y,y_pred)))
print('Mean Squared error of all column: ' + str(metrics.mean_squared_error(y,y_predall)))
#print(metrics.mean_squared_error(y,y_pred))
#print(metrics.mean_squared_error(y,y_predall))
