# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:01:03 2023

@author: user
"""

import pandas as pd 
from sklearn import linear_model 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy import ravel, savez

# Загрузка данных из файлов Excel
df1 = pd.read_excel("X_bp.xlsx")
df1 = df1.drop('Unnamed: 0', axis=1) # удалите первый столбец; он нам не нужен

df2 = pd.read_excel("X_nup.xlsx")
df2 = df2.drop('Unnamed: 0', axis=1) # удалите первый столбец; он нам не нужен

# INNER join
df_joined = df1.join(df2, how='inner')

# Переименуйте заголовки, как в Excel, для упрощения манипуляций
df_joined.set_axis([chr(65+i) for i in range(13)], axis='columns', inplace=True)

feature_columns = [chr(65+i) for i in range(13) if chr(65+i) not in ("H", "I", "K")]

for K in [ 90, 0 ] :
    
    # Фильтровать данные на основе значений столбца K.
    df = df_joined[lambda df: df['K'] == K].drop('K', axis=1)
    
    # Изолировать столбцы объектов = все столбцы, КРОМЕ столбцов "H" и "I"
    featureData = df.filter(feature_columns)
    
    # Масштабируйте данные. 
    scaler = StandardScaler()
    featureData = scaler.fit_transform(featureData.values)
    
    savez("SGD_scaler_%d.npz" % K,  mean=scaler.mean_, scale=scaler.scale_)

    for target_column in [ 'H', 'I' ]:
        
        # Изолируйте целевой столбец
        targetData = df.filter([target_column])
     

        featureData_training, featureData_test, \
        targetData_training, targetData_test = train_test_split (featureData, 
                                                                 targetData, 
                                                                 test_size=30,
                                                                 random_state=42)
        # Закомментируйте следующие две строки, если вы хотите 
        # отделить тестовые выборки от общей совокупности
        # featureData_training = featureData_test = featureData
        # targetData_training  = targetData_test  = targetData
        
        print("\n\n=== Обучающий алгоритм стохастического градиентного спуска (target = %s , K = %d) ===" % (target_column, K)) 
            
        # Создать алгоритм/движок регрессии SGD
        regr = linear_model.SGDRegressor(verbose=0, tol=1e-6)
        
        # Обучите алгоритм / движок, используя обучающие наборы
        regr.fit(featureData_training,  ravel(targetData_training))
        
        print("\n===================\nЗначение R2: %.2f (обучение образцы) %.2f (тестовые образцы) ; итерации: %d \n===================" %
               (regr.score(featureData_training, targetData_training),
                regr.score(featureData_test, targetData_test),
                regr.n_iter_))
        
        savez("SGD_%d_%s.npz" % (K, target_column), coefs=regr.coef_, intercept=regr.intercept_)
        