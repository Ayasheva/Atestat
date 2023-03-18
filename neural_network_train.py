# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:31:29 2023

@author: user
"""

import pandas as pd 
from sklearn import neural_network 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy import ravel
import pickle

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

config = { 
      0 : { 'layers_count' : 25,
            'layer_size'   : 46  } ,
    90 : { 'layers_count' : 24,
            'layer_size'   : 43  } ,
    'activation' : 'relu'
}


# config = { 
#       0 : { 'layers_count' : 8,
#             'layer_size'   : 100  } ,
#     90 : { 'layers_count'  : 8,
#             'layer_size'   : 100 } ,
#     'activation' : 'logistic'
# }

for K in [ 90, 0 ] :
    
    # Фильтровать данные на основе значений столбца K.
    df = df_joined[lambda df: df['K'] == K].drop('K', axis=1)
    
    # Изолировать столбцы объектов = все столбцы, КРОМЕ столбцов "A"
    featureData = df.filter(feature_columns)
    
    # Масштабируйте данные. 
    scaler = StandardScaler()
    featureData = scaler.fit_transform(featureData.values)
    
    file = open("NN_scaler_%d_A.pckl" % K, "wb")
    pickle.dump(scaler, file)
    file.close()
    
    # Изолируйте целевой столбец
    targetData = df.filter(['A'])
 
    featureData_training, featureData_test, \
    targetData_training, targetData_test = train_test_split (featureData, 
                                                             targetData, 
                                                             test_size=30,
                                                             random_state=42)
    
    # Закомментируйте следующие две строки, если вы хотите 
    # отделить тестовые выборки от общей совокупности
    # featureData_training = featureData_test = featureData
    # targetData_training  = targetData_test  = targetData
    
    print("\n=== Обучение алгоритму стохастического градиентного спуска (K = %d) ===" % K)
        
    # Создать алгоритм/движок регрессии
    hidden_layer_size=config[K]['layer_size']
    hidden_layer_count=config[K]['layers_count']                    
    regr = neural_network.MLPRegressor(random_state=1, 
                         activation=config['activation'],
                         verbose=False,
                         hidden_layer_sizes=[hidden_layer_size 
                                             for i in range(hidden_layer_count)],
                         solver='adam', 
                         batch_size = 20,
                         tol=1e-6, 
                         n_iter_no_change=50, 
                         max_iter=int(1e4))
    
    # Обучите алгоритм / движок, используя обучающие наборы
    regr.fit(featureData_training,  ravel(targetData_training))
    
    print("\nЗначение R2: %.2f (обучение образцы) %.2f (тестовые образцы) ; итерации: %d" %
            (regr.score(featureData_training, targetData_training),
             regr.score(featureData_test, targetData_test),
             regr.n_iter_))

    # Сохраните нейронную сеть на диск
    file = open("NN_%d_A.pckl" % K, "wb")
    pickle.dump(regr, file)
    file.close()
    