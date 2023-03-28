# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:36:52 2023

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy

# Загрузка данных из файлов Excel
df1 = pd.read_excel("X_bp.xlsx")
df1 = df1.drop('Unnamed: 0', axis=1) # удалите первый столбец; он нам не нужен

df2 = pd.read_excel("X_nup.xlsx")
df2 = df2.drop('Unnamed: 0', axis=1) # удалите первый столбец; он нам не нужен

#  INNER join
df = df1.join(df2, how='inner')

# Переименуйте заголовки, как в Excel, для упрощения манипуляций
df.set_axis([chr(65+i) for i in range(13)], axis='columns', inplace=True)

#df = df[lambda df: df['K'] == 0]

plt.figure("Raw data distribution")
for i in range(13):
    plt.subplot(3,5, i+1)
    plt.hist(df[chr(65+i)])
    plt.grid(True)
plt.show()

scaler = StandardScaler()
scaled = numpy.transpose(scaler.fit_transform(df))

plt.figure("Scaled data distribution")
for i in range(13):
    plt.subplot(3,5, i+1)
    plt.hist(scaled[i])
    plt.grid(True)
plt.show()

