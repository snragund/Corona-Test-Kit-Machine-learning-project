# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:29:19 2020

@author: gundawar
"""
#%% Import Library
import pandas as pd
import numpy as np
import pickle 
from sklearn.linear_model import LogisticRegression

def data_split(data,ratio):
#%% Train Test Splliting
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "main":
    
    #%% Reading Data
    df = pd.read_csv('Corona_data.csv')
    df.head()
    #%%
    df.info()

    #%%
    df['DiffBreath'].value_counts()

    #%%
    df.describe()

    #%% Split Test-Train Data
    train, test = data_split(df, 0.2)

    #%%
    X_train = train[['Fever', 'BodyPain', 'Age', 'RunnyNose', 'DiffBreath','SoreThroat','Cough','Tiredness','RecentTravel']].to_numpy()
    X_test = test[['Fever', 'BodyPain', 'Age', 'RunnyNose', 'DiffBreath','SoreThroat','Cough','Tiredness','RecentTravel']].to_numpy()

    #%%
    Y_train = train[['InfectionProb']].to_numpy().reshape(2400,)
    Y_test = test[['InfectionProb']].to_numpy().reshape(600,)

    #%%
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    file = open('model.sav', 'wb')
    pickle.dump(clf, file)
    file.close

    #%% code for inference
    inputFeatures = [100,1,22,1,1,1,1,1,1]
    InfectionProbability = clf.predict_proba([inputFeatures])[0][1]
    print(InfectionProbability)
