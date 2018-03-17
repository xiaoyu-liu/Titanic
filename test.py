##------------------- Import Packages -------------------------
##-------------------------------------------------------------

# Basic packages for database and for scientific computing
import os
import pandas as pd
import numpy as np
# import random as rnd

# Visualization tool
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

##--------------------- Input Dataset -------------------------
##-------------------------------------------------------------
os.chdir("C:\\Summer\\Databases by Python - Coursera\\Kaggle Project\\Titanic-180109")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

##---------- Check the information of dataset -----------------
##-------------------------------------------------------------
train_df.info()
test_df.info()
#**************************************************************
# Summary of dataset through abservations:
#   1. For training set, only Age, Cabin and Embarked have 
#      missing value; while; for testing set, Age, Cabin and Fare
#      have missing value.
#   2. Data Type:
#      Int    - PassengerId, Pclass, SibSp, Parch, Survived
#      Float  - Fare, Age
#      object - Name, Sex, Ticket, Cabin, Embarked
#   3. Let Survived be response variable, then another 11 features are
#      predictors.
#   4. Cabin has 77.11% missing value for training group, and 78.23% 
#      for testing group. Therefore, I consider to eliminate this 
#      features based on its absence.
#**************************************************************

##---------------- Create some new features -------------------
##---------in order to reduce the dimension of predictor-------

# Create a new feature 'FamSize'
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in combine:
    dataset['FamSize'] = 0
    dataset.loc[(dataset['FamilySize'] > 0) & (dataset['FamilySize'] < 5), 'FamSize'] = 1
    dataset.loc[dataset['FamilySize'] >= 5, 'FamSize'] = 2
    ## 0: single
    ## 1: small family
    ## 2: big family
#***************************************************************
# 'FamSize' represents the size of family in this travels, after 
# building this, 'SibSp' and 'Parch' can be eliminated.
#***************************************************************

# Create a new feature 'FamName'
for dataset in combine:
    dataset['FamName'] = dataset.Name.str.extract('([A-Za-z]+)\,', expand=False)
#***************************************************************
# 'FamName' represents the family name of travelers, after 
# building this, 'Name' can be eliminated.
#
# On the other hand, I will use this variable in building 
# variable 'FamwithChild' which will contain information from 
# 'FamName', 'FamSize', as well as 'Age'.
#***************************************************************

# Create a new feature 'Title'
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
       ## From the chart shown above, I didn't seen any useful information from
       ## title except indication of age and sex from title.
#***************************************************************
# 'Title' represents the titles of travelers, after building this,
# 'Name' can be eliminated.
#
# On the other hand, I will use this variable in prediction 
# missing value of age for both training and testing sets.
#***************************************************************

##---------------- Deal with missing values -------------------
##-------------------------------------------------------------

# For Age (both in Train and Test):
#     Step 1: Build another dataframe, it contains the mode of 
#             age under each Title.
AveAgeTitle = pd.DataFrame({'Title':['Master','Miss','Mr','Mrs','Rare']})
for title in AveAgeTitle['Title']:
    AveAgeTitle.loc[AveAgeTitle['Title']==title,'AveAge'] \
    = train_df[train_df['Title'] == title]['Age'].dropna().mode()[0]
#     Step 2: Fill in missing value with mode with identical Title
for dataset in combine:
    for title in AveAgeTitle.Title:
        FillAge = AveAgeTitle.loc[AveAgeTitle['Title']==title , 'AveAge'].mean()
        dataset.loc[(dataset.Age.isnull()) & (dataset['Title'] == title), 'Age'] = FillAge

# For Embarked (in Train):
#     Since Embarked contains categorical variables,
#     I decided to use the mode of it to fill in missing values.
train_df.fillna(train_df.Embarked.dropna().mode()[0])

# For Fare (in Test):
#     Since Fare contains continous variables,
#     I decided to use median of it to fill in missing values.
#     To be noticed, using mean is also meaningful to try, but 
#     result will not change to much. (median is not so different with mean)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

##--------------------- Transfer Dataset ----------------------
##-------------------------------------------------------------

# Transfer continous data into limited groups
# For Age:
#         0: infants
#         1: children
#         2: teenagers
#         3: adults
#         4: middle ages
#         5: aged people   
for dataset in combine:    
    dataset.loc[ dataset['Age'].dropna() <= 2, 'Age'] = 0
    dataset.loc[(dataset['Age'].dropna() > 2) & (dataset['Age'].dropna() <= 12), 'Age'] = 1
    dataset.loc[(dataset['Age'].dropna() > 12) & (dataset['Age'].dropna() <= 18), 'Age'] = 2
    dataset.loc[(dataset['Age'].dropna() > 18) & (dataset['Age'].dropna() <= 40), 'Age'] = 3
    dataset.loc[(dataset['Age'].dropna() > 40) & (dataset['Age'].dropna() <= 60), 'Age'] = 4
    dataset.loc[ dataset['Age'].dropna() > 60, 'Age'] = 5
    dataset['Age'] = dataset['Age'].astype(int)
# For Fare:
#          0: really low fare      - (0, 7.75]
#          1: low fare             - (7.75, 8.05]
#          2: medium low fare      - (8.05, 12.48]
#          3: medium fare          - (12.48, 19.26]
#          4: medium high low fare - (19.26, 27.9]
#          5: high fare            - (27.9, 56.9]
#          6: really high fare     - 56.9+
train_df['FareBand'] = pd.qcut(train_df['Fare'], 7)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.75, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.75) & (dataset['Fare'] <= 8.05), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 8.05) & (dataset['Fare'] <= 12.48), 'Fare']  = 2
    dataset.loc[(dataset['Fare'] > 12.48) & (dataset['Fare'] <= 19.26), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 19.26) & (dataset['Fare'] <= 27.9), 'Fare']   = 4
    dataset.loc[(dataset['Fare'] > 27.9) & (dataset['Fare'] <= 56.9), 'Fare'] = 5
    dataset.loc[ dataset['Fare'] > 56.9, 'Fare'] = 6
    dataset['Fare'] = dataset['Fare'].astype(int)

# Transfer string data into int
# for sex:
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# for embarked   
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'Q': 2, 'S': 1, 'C': 0} ).astype(int)


##-------------------- Create a feature -----------------------
##--------- based on trasfered Age, FamName and FamSize -------

# Create a new feature 'FamwithChild':
#                                       0: no child in family in this trip
#                                       1: infant with family in this trip
#                                       2: child with family in this trip
FamNameGroup = train_df[['FamName','Survived']].groupby(['FamName'],as_index=False).count()
FamNameGroup = FamNameGroup[FamNameGroup['Survived']>1]
FamName = FamNameGroup['FamName'].tolist()
train_df['FamwithChild'] = 0
for lastname in FamName:
    train_df.loc[(train_df['FamName'] == lastname) & (train_df['FamSize'] > 0)\
                 & (train_df['Age'].min() == 0), 'FamwithChild'] = 1
    train_df.loc[(train_df['FamName'] == lastname) & (train_df['FamSize'] > 0)\
                 & (train_df['Age'].min() == 1), 'FamwithChild'] = 2
        
FamNameGroup2 = test_df[['FamName','Age']].groupby(['FamName'],as_index=False).count()
FamNameGroup2 = FamNameGroup2[FamNameGroup2['Age']>1]
FamName2 = FamNameGroup2['FamName'].tolist()
test_df['FamwithChild'] = 0
for lastname in FamName2:
    test_df.loc[(test_df['FamName'] == lastname) & (test_df['FamSize'] > 0)\
                 & (train_df['Age'].min() == 0), 'FamwithChild'] = 1
    test_df.loc[(test_df['FamName'] == lastname) & (test_df['FamSize'] > 0)\
                 & (train_df['Age'].min() == 1), 'FamwithChild'] = 2

##-------------------- Drop useless variables -----------------------
##-------------------------------------------------------------------
train_df = train_df.drop(['Ticket', 'Cabin', 'Name', 'Title', 'PassengerId',\
                  'FamilySize', 'Parch', 'SibSp','FareBand', 'FamName'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Name', 'Title', 'PassengerId',\
                  'FamilySize', 'Parch', 'SibSp','FamName'], axis=1)

