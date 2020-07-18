# Titanic Survival Analysis 

In this chapter, the writter use the legendary dataset for many of data analyst. Here, the Titanic survival classification as the feature were built by machine learning calculation. This dataset was collected from Kaggle. For the begining, let me introduce the path of the project:

1. Exploratory Data Analysis
2. Feature Engineering
3. Developing model machine learning

The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

<h2>RMS Titanic</h2>

The RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after it collided with an iceberg during its maiden voyage from Southampton to New York City. There were an estimated 2,224 passengers and crew aboard the ship, and more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history. The RMS Titanic was the largest ship afloat at the time it entered service and was the second of three Olympic-class ocean liners operated by the White Star Line. The Titanic was built by the Harland and Wolff shipyard in Belfast. Thomas Andrews, her architect, died in the disaster.


<img 
src="https://miro.medium.com/max/450/0*l5aRNzEo1QNsVn7u.jpg"/>

<h3>Import Data Set dan Concat Data Frame</h3>
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import string
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

```python
def concat_df(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv('titanic_test.csv')
df_all = concat_df(df_train, df_test)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set'

dfs = [df_train, df_test]

print('Number of Training Examples = {0}'.format(df_train.shape[0]))
print('Number of Test Examples = {0}\n'.format(df_test.shape[0]))
print('Training X Shape = {0}'.format([df_train.shape]))
print('Training y Shape = {0}\n'.format([df_train['Survived'].shape[0]]))
print('Test X Shape = {0}'.format([df_test.shape]))
print('Test y Shape = {0}\n'.format([df_test.shape[0]]))
print(df_train.columns)
print(df_test.columns)
```
Number of Training Examples = 891
Number of Test Examples = 418

Training X Shape = [(891, 12)]
Training y Shape = [891]

Test X Shape = [(418, 11)]
Test y Shape = [418]

Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

<h2>Exploratory Data Analysis</h2>
PassengerId adalah id pada row, maka tidak ada pengaruh terhadap target yang dicari
Survived adalah target yang akan diprediksi, nilai 0 = Not Survived dan nilai 1 = Survived
Pclass (Passenger Class) adalah kategori level sosial ekonomi penumpang dengan nilai (1, 2 atau 3):
1 = Upper Class
2 = Middle Class
3 = Lower Class
Name, Sex dan Age merupakan data self-explanatory
SibSp adalah jumlah saudara dari penumpang
Parch adalah jumlah Orang Tua dan anak dari penumpang
Ticket adalah jumlah tiket penumpang
Fare adalah tarif yang di kenakan kepada penumpang
Cabin adalah nomor kabin penumpang
Embarked adalah pelabuhan pemberangkatan ada 3 pelabuhan (C, Q atau S):
C = Cherbourg
Q = Queenstown
S = Southampton


