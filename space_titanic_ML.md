---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Subject: space Titanic prediction 
# Link: https://www.kaggle.com/competitions/spaceship-titanic/data?select=sample_submission.csv
# Author: Pierre Mulliez 
# Date start: 28/11/2022
# Description: classify passengers as transported successfully (TRUE) or not (FALSE)

```python
# import packages 
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```

```python
# import data and print summaries
titanic_df = pd.read_csv("data/train.csv")
print(titanic_df.head(5))
print('------------------')
print('dataframe shape: {}'.format(titanic_df.shape))
print('------------------')
print(titanic_df.describe())
```

```python
# baseline model without feature engineering all TRUE 
test = [1 for count in range(0,len(titanic_df))]
no_feature_eng_error = accuracy_score(titanic_df.loc[:,'Transported'].factorize()[0],test)
print('baseline accuracy without feature engineering is {}'.format(round(no_feature_eng_error,2)))
```

```python
# null values 
print(titanic_df.isnull().sum())
#titanic_df = titanic_df.dropna()
titanic_df.loc[titanic_df['Destination'].isnull() == True,:].head(10)
```

```python
# train test split
# drop unessesary collumns
y = titanic_df.loc[:,'Transported']
X = titanic_df.loc[:,(titanic_df.columns != 'Transported') & (titanic_df.columns != 'Name')]

#target encoding
def encode(X):
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()
    return X
X = encode(X)
```

```python
# scaling
def scale(X):
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_scaled
X_scaled = scale(X)
```

```python
### null values 

# values to predict as keys, independent varaibles as values 
xy_nulls = {'Age':('RoomService','Spa','FoodCourt'),
            'VIP':('Age','RoomService','Spa'),
             'HomePlanet':('Age','RoomService','VIP'),
             'VRDeck':('Age','RoomService','VIP','HomePlanet'),
            'ShoppingMall':('RoomService','Spa','FoodCourt','VIP','Age')
               }
ys = list(xy_nulls.keys())

def replace_nulls(data,train_variables,collumn_predict):
    
    # if nulls in train variable mean 
    for train_var in train_variables:
        data.loc[data[train_var].isnull() == True,train_var] = data[train_var].mean()
    
    # get variable to predict
    data_pred = data.loc[data[collumn_predict].isnull() == True,:]
    data_train = data.loc[data[collumn_predict].isnull() == False,:]
    
    if len(data_pred) == 0:
        #escape if no null values
        return data
    
    X = data_train.loc[:,train_variables]
    y = data_train[collumn_predict]
    
    reg = LinearRegression().fit(X, y)
    
    #replace null values 
    data.loc[data[collumn_predict].isnull() == True,collumn_predict] = reg.predict(data_pred.loc[:,train_variables])
    return data

X_cleaned = X_scaled.copy()
for ys_unit in ys:
    X_cleaned = replace_nulls(X_cleaned, xy_nulls[ys_unit],ys_unit)

print(X_cleaned.isnull().sum())
```

```python
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.7, random_state=42)
```

```python
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
first_model_error = accuracy_score(y_test,predictions)
print('Random forest accuracy:  {}'.format(round(first_model_error,2)))
```
