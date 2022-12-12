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
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
```

```python
# import data and print summaries
titanic_df = pd.read_csv("data/train.csv")
raw_Kaggle = pd.read_csv("data/test.csv")
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
titanic_df.loc[titanic_df['Destination'].isnull() == True,:].head(15)
```

```python
# train test split
# drop unessesary collumns
y = titanic_df.loc[:,'Transported']
X = titanic_df.loc[:,(titanic_df.columns != 'Transported')]
```

```python
#target encoding
def encode(X):
    # cabin work 
    X['cabin_1letter'] = [X.loc[nber,'Cabin'].split('/')[0] if type(X.loc[nber,'Cabin']) == str else next for nber in range(0,len(X['Cabin']))]
    X['cabin_number'] = [int(X.loc[nber,'Cabin'].split('/')[1]) if type(X.loc[nber,'Cabin']) == str else 0 for nber in range(0,len(X['Cabin']))]
    X['cabin_3letter'] = [X.loc[nber,'Cabin'].split('/')[2] if type(X.loc[nber,'Cabin']) == str else next for nber in range(0,len(X['Cabin'])) ]
    X = X.drop('Cabin',axis = 1)
   #id work 
    X['id_1'] = [int(X.loc[nber,'PassengerId'].split('_')[0]) if type(X.loc[nber,'PassengerId']) == str else 0 for nber in range(0,len(X['PassengerId']))]
    X['id_2'] = [int(X.loc[nber,'PassengerId'].split('_')[1]) if type(X.loc[nber,'PassengerId']) == str else 0 for nber in range(0,len(X['PassengerId']))]
    X = X.drop('PassengerId',axis = 1)
    
    #family names 
    X['FamilyName'] = [X.loc[nber,'Name'].split(' ')[0] if type(X.loc[nber,'Name']) == str else next for nber in range(0,len(X['Name']))]
    X = X.drop('Name',axis = 1)
    
    #boolean to int
    for colname in ('CryoSleep','VIP'):
        X[colname] = [int(1) if X.loc[el,colname] == 'True' else 0 for el in range(0,len(X[colname]))]
        
    #factorize the rest
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()
        
    return X
```

```python
def kmean_pipe(X):
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()
    for colname in X.select_dtypes("float"):
        X[colname] = scale(X[colname])
    return X
```

```python
### null values 

# values to predict as keys, independent varaibles as values 
xy_nulls = {'Age':('RoomService','Spa','FoodCourt'),
            'VIP':('Age','RoomService','Spa'),
             'HomePlanet':('Age','RoomService','VIP'),
            'Destination':('HomePlanet','Age'),
            'CryoSleep':('HomePlanet', 'Destination','Age'),
             'VRDeck':('Age','RoomService','VIP'),
            'ShoppingMall':('RoomService','Spa','FoodCourt','VIP','Age')
               }
ys = list(xy_nulls.keys())

def replace_nulls(data,train_variables,collumn_predict):
    
    # if nulls in train variable mean 
    for train_var in train_variables:
        if  type(data.loc[data[train_var].isnull() == False,train_var][0] ) == str:
            data.loc[data[train_var].isnull() == True,train_var] = data[train_var].mode()
        else:
            data.loc[data[train_var].isnull() == True,train_var] = data[train_var].mean()
    
    # get variable to predict
    data_pred = data.loc[data[collumn_predict].isnull() == True,:]
    data_train = data.loc[data[collumn_predict].isnull() == False,:]
    
    if len(data_pred) == 0:
        #escape if no null values
        return data
    
    X = data_train.loc[:,train_variables]
    y = data_train[collumn_predict]
    X = kmean_pipe(X)
    if  type(data_train.loc[0,collumn_predict]) == str:
        null_prediction = KMeans(n_clusters=6).fit(X, y)
    else:
        null_prediction = LinearRegression().fit(X, y)
    
    data_pred = kmean_pipe(data_pred.loc[:,train_variables].copy())
    #replace null values 
    data.loc[data[collumn_predict].isnull() == True,collumn_predict] = null_prediction.predict(data_pred.loc[:,train_variables])
    return data
```

```python
# scaling
def scale(X):
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_scaled
```

```python
def clustering(X,col = ('Age','VIP','cabin_1letter')):
    kmeans = KMeans(n_clusters=6)
    data_pred = kmean_pipe(X.loc[:,col].copy())
    X["Cluster"] = kmeans.fit_predict(data_pred.loc[:,col])
    X["Cluster"] = X["Cluster"].astype("category")
    return X
```

```python
X = encode(X)
for ys_unit in ys:
    X_cleaned = replace_nulls(X, xy_nulls[ys_unit],ys_unit)
X_enriched = clustering(X_cleaned)
```

```python
X_train, X_test, y_train, y_test = train_test_split(X_enriched, y, test_size=0.3, random_state=42)
```

```python
#grid search for optimal parameters
param_grid = { 
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2,3,4,6],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv= 4)
```

```python
CV_rfc.fit(X_train,y_train)
CV_rfc.best_params_
```

```python
bestforest = RandomForestClassifier(criterion  = CV_rfc.best_params_['criterion'], 
                                    max_depth = CV_rfc.best_params_['max_depth'], 
                                    max_features = CV_rfc.best_params_['max_features'])
bestforest.fit(X_train,y_train)
predictions = bestforest.predict(X_test)
first_model_error = accuracy_score(y_test,predictions)
print('Random forest accuracy:  {}'.format(round(first_model_error,2)))
```

```python
titanic_df_Kaggle = encode(raw_Kaggle)
for ys_unit in ys:
    titanic_df_Kaggle = replace_nulls(titanic_df_Kaggle, xy_nulls[ys_unit],ys_unit)
titanic_df_Kaggle_clustered = clustering(titanic_df_Kaggle)
titanic_df_Kaggle_submit = bestforest.predict(titanic_df_Kaggle_clustered)
```

```python
#final result 
submit = pd.DataFrame(raw_Kaggle.loc[:,'PassengerId'])
submit['Transported'] = titanic_df_Kaggle_submit
print(submit.head(15))
submit.to_csv('./data/submit_{}.csv'.format(datetime.today().strftime("%d_%m_%Y")),index=False)
```

```python
# last score 0.77 
```
