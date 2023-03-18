# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:55:38 2023

@author: user
"""

import numpy

scaler = dict()
predictor = { 90 : dict() , 
               0 : dict() }

for K in [ 90, 0 ] :
    scaler_ = numpy.load("SGD_scaler_%d.npz" % K)
    scaler[K] = { 'mean'  : scaler_['mean'],
                  'scale' : scaler_['scale']}
    for target_column in [ 'H' , 'I' ]:
        predictor_ = numpy.load("SGD_%d_%s.npz" % (K, target_column))
        predictor[K][target_column] = { 
            'coefs'      : predictor_['coefs'] ,
            'intercept'  : predictor_['intercept']
        }

n_features = len(predictor[0]['H']['coefs'])+1

help_message = '''
Параметры следует вводить в таком порядке:

 1.	Соотношение матрица-наполнитель
 2.	Плотность, кг/м3	
 3.	модуль упругости, ГПа	
 4.	Количество отвердителя, м.%	
 5.	Содержание эпоксидных групп,%_2	
 6.	Температура вспышки, ᵒС	
 7.	Поверхностная плотность, г/м2	
 8.	Потребление смолы, г/м2
 9.	Угол нашивки, град	(ᵒ)
10.	Шаг нашивки	
11.	Плотность нашивки

'''

print(help_message)

while True:
    user_input = input('Введите 11 параметров (через пробел): ')
    
    if not user_input:
        break
    
    features = numpy.fromstring(user_input, dtype=float, sep=' ')
    # features = 	numpy.array([ 
    #               2.49991792788794,	1942.59577668332,	901.51994673357,	
    #               146.252207766867,	23.0817574798037,	351.231873967312,	
    #               864.725483802792,	226.222760364565,	
    #               90, 5.0, 47.0 ])

    if len(features) != n_features :
        print ("\n*** Неверное количество входных параметров! "
               "Получил %d. Ожидаемый %d ***\n" %
              (len(features), n_features))
        continue
    K=int(features[8])
    features = (numpy.delete(features, 8) - scaler[K]['mean'])/scaler[K]['scale']
    
    print("Модуль упругости при растяжении = %.2f ГПа" % 
          ( numpy.dot(features, predictor[K]['H']['coefs']) + 
            predictor[K]['H']['intercept'][0]) ) 
    
    print("Прочность при растяжении = %.2f МПа\n" % 
          ( numpy.dot(features, predictor[K]['I']['coefs']) + 
            predictor[K]['I']['intercept'][0]) )
    
print('''
*******************
* Конец программа *
*******************''')