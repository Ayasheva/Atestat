# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:31:29 2023

@author: user
"""

import pandas as pd 
from sklearn import neural_network 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Загрузка данных из файлов Excel
df1 = pd.read_excel("X_bp.xlsx")
df1 = df1.drop('Unnamed: 0', axis=1) # удалите первый столбец; он нам не нужен

df2 = pd.read_excel("X_nup.xlsx")
df2 = df2.drop('Unnamed: 0', axis=1) # удалите первый столбец; он нам не нужен

# INNER join
df_joined = df1.join(df2, how='inner')

# Переименуйте заголовки, как в Excel, для упрощения манипуляций
df_joined.set_axis([chr(65+i) for i in range(13)], axis='columns', inplace=True)

feature_columns = [chr(65+i) for i in range(13) if chr(65+i) not in ("A", "K")]

fp = open("Neural-Network-Training.log.txt", "wt", encoding='utf-8')

msg = "\n".join(
[    
"R2 score\n",
"+-------------------+------------------+--------------+-------------+---------------+",
"| обучение образцы  | тестовые образцы | Угол нашивки |  Layer Size | Number Layers |",
"+-------------------+------------------+--------------+-------------+---------------+",
])

print(msg)
fp.write(msg+"\n")
fp.close()

for layer_size in range(20, 50):
    for number_layers in range (10, 60) :
        for K in [ 90, 0 ] :
            
            # Фильтровать данные на основе значений столбца K.
            df = df_joined[lambda df: df['K'] == K].drop('K', axis=1)
            
            # Изолировать столбцы объектов = все столбцы, КРОМЕ столбцов "A"
            featureData = df.filter(feature_columns)
            
            # Масштабируйте данные. 
            scaler = StandardScaler()
            featureData = scaler.fit_transform(featureData.values)
            
            target_column = 'A'
            
            # Изолируйте целевой столбец
            targetData = df.filter([target_column])
         
        
            featureData_training, featureData_test, \
            targetData_training, targetData_test = train_test_split (featureData, 
                                                                     targetData, 
                                                                     test_size=30,
                                                                     random_state=42
                                                                     )
            # Закомментируйте следующие две строки, если вы хотите 
            # отделить тестовые выборки от общей совокупности
            # featureData_training = featureData_test = featureData
            # targetData_training  = targetData_test  = targetData
            
            # print("\n\n=== Обучающий алгоритм стохастического градиентного спуска (target = %s , K = %d) ===" % (target_column, K)) 
                
            # Создать алгоритм/движок регрессии 
            regr = neural_network.MLPRegressor(random_state=1, 
                                 activation='relu', 
                                 verbose=False,
                                 hidden_layer_sizes=[layer_size for i in range(number_layers)], # , 30,40,
                                 solver='adam', 
                                 batch_size = 20, 
                                 tol=1e-6, 
                                 n_iter_no_change=50, 
                                 max_iter=int(1e4))
            
            # Обучите алгоритм / движок, используя обучающие наборы
            regr.fit(featureData_training,  ravel(targetData_training))
            
            msg = "|       %+6.2f      |      %+6.2f      |       %4d   |      %4d   |     %4d      |" % \
                   (regr.score(featureData_training, targetData_training),
                    regr.score(featureData_test, targetData_test),
                    K, layer_size, number_layers)

            print(msg)
            fp = open("Neural-Network-Training.log.txt", "at", encoding='utf-8')
            fp.write(msg+"\n")
            fp.close()
            
