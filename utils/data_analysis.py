# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:31:29 2023

@author: user
"""

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
import numpy as np
import seaborn as sns
warnings.filterwarnings("ignore")

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


for iter_count, K in enumerate([ 90, 0 ]) :
    
    # Фильтровать данные на основе значений столбца K.
    df = df_joined[lambda df: df['K'] == K].drop('K', axis=1)

    
    # Изолировать столбцы объектов = все столбцы, КРОМЕ столбцов "A"
    featureData = df.filter(feature_columns)
    
    # Масштабируйте данные. 
    scaler = StandardScaler()
    featureData = scaler.fit_transform(featureData.values)
      
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
    
    if True:
        corr_matrix = df.corr().round(2)
        plt.subplot(1,2, iter_count+1)
        sns.heatmap(corr_matrix, annot=False)
        plt.title('K = %d' % K)
        plt.show()
    
    
    if False:
        pca = PCA()
        x_3d = pca.fit_transform(featureData_training) 
        plt.figure(figsize=(8, 6))
        plt.bar(range(11), pca.explained_variance_, alpha=0.5, align='center', label='individual variance')
        plt.legend()
        plt.ylabel('Variance ratio')
        plt.xlabel('Principal components')
        plt.title('K = %d' % K)
        plt.show()
    

    if False:
        colors = []
        x_3d = PCA().fit_transform(featureData_training)
        for c in np.random.rand(11, 3) :      
            colors.append("#%02x%02x%02x" % (int(255*c[0]), int(c[1]*255), int(255*c[2])))
        plt.figure(figsize=(8,6))
        plt.title('K = %d' % K)
        for i in range(1,11):
            plt.subplot(5,2, i)
            plt.scatter(x_3d[:,0], x_3d[:,i], color = colors[i])
            plt.grid(True)
        plt.show()
        
    
    