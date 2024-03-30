# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:38:23 2024

@author: Shooq
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#1read data
df=pd.read_csv("C:/Users/Shooq/Downloads/tit/titanic.csv")

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
#print(df.head)
# 2 Data Summary
def print_statistical_summaries(df):
    
    avg_values = df.mean()
    print("Average values:")
    print(avg_values)
    
    
    most_frequent_values = df.mode().iloc[0]
    print("\nMost frequent values:")
    print(most_frequent_values)

# Call the function with the sample dataset
print_statistical_summaries(df)
#3 handle null values
print("age summruze is:",df.Age.mean)

df.isnull().sum()
df.Age = df.Age.fillna(df.Age.mean())

#encode
dummies = pd.get_dummies(df['Sex'])
merged = pd.concat([df,dummies],axis='columns')
x = merged.drop(["Sex"],axis='columns')
le = LabelEncoder()
dfle = df
dfle.Sex = le.fit_transform(dfle.Sex)
ct = ColumnTransformer(transformers =[('encoder', OneHotEncoder(), [2])], remainder = 'passthrough')
x=df.drop(['Survived'],axis='columns')
x = ct.fit_transform(x)
x=df.drop(['Survived'],axis='columns')
print(x)
print(x.head)