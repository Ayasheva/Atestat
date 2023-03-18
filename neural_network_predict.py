# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:55:38 2023

@author: user
"""

import pickle
import numpy

scaler = dict()
predictor = dict()

# Загрузите нейронную сеть с диска
for K in [ 90, 0 ] :
    file = open("NN_scaler_%d_A.pckl" % K, "rb")
    scaler[K] = pickle.load(file)
    file.close()
    
    file = open("NN_%d_A.pckl" % K, "rb")
    predictor[K] = pickle.load(file)
    file.close()
    
print('''
Параметры следует вводить в таком порядке:

 1.	Плотность, кг/м3	
 2.	модуль упругости, ГПа	
 3.	Количество отвердителя, м.%	
 4.	Содержание эпоксидных групп,%_2	
 5.	Температура вспышки, ᵒС	
 6.	Поверхностная плотность, г/м2	
 7.	Модуль упругости при растяжении, ГПа	
 8.	Прочность при растяжении, МПа	
 9.	Потребление смолы, г/м2
10.	Угол нашивки, град (ᵒ)
11.	Шаг нашивки	
12.	Плотность нашивки

''')

n_input_columns = 12

while True:
    user_input = input('Введите 12 параметров (через пробел): ')
    
    if not user_input:
        break
    
    features = numpy.fromstring(user_input, dtype=float, sep=' ')

    if len(features) != n_input_columns :
        print ("\n*** Неверное количество входных параметров! "
               "Получил %d. Ожидаемый %d ***\n" % (len(features), n_input_columns))
        continue
    
    K=int(features[9]) ; assert K==90 or K==0
    
    features = scaler[K].transform(numpy.delete(features, 9).reshape(1, -1))
    
    print("Соотношение матрица-наполнитель = %.2f" % predictor[K].predict(features))
    

print('''
*******************
* Конец программа *
*******************''')