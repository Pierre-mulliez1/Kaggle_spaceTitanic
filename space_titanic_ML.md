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
from sklearn.model_selection import train_test_split
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
print('baseline error without feature engineering is {}'.format(round(no_feature_eng_error,2)))
```

```python
# train test split
y = titanic_df.loc[:,'Transported']
X = titanic_df.pop('Transported')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
```

```python
# scaling
def scale(X):
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_scaled
X_scaled = scale(X_train)
```
