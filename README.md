﻿Испытательный проект
--------------------

Все "*_train.py" программы требуют `X_bp.xlsx` и `X_nup.xlsx` Файлы данных Excel должны присутствовать в текущем каталоге. Они являются основным источником данных (обучающих выборок) для алгоритмов обучения.

Для всех скриптов Python требуются библиотеки **Pandas** и **SciKit-Learn** для установки.
Для некоторых служебных скриптов также требуется наличие библиотеки Matplot.

Основные программы:
------------------

1. **linear_regression_sgd_train.py**

	* Цель: Реализация алгоритма стохастического градиентного спуска для модели линейной регрессии. Он считывает данные из файлов Excel, а затем обучает алгоритм. По завершении он сохраняет состояние алгоритма во внешних файлах, включая масштабатор, используемый для нормализации данных.
	* Использование: `py linear_regression_sgd_train.py`
	* Ввод:
         * X_bp.xlsx 
         * X_nup.xlsx 
	* Вывод:
         * SGD_scaler_{0/90}.npz 
         * SGD_{0/90}_{H|I}.npz 

2. **linear_regression_sgd_predict.py** 
	* Цель: Предсказатель линейной регрессии на основе алгоритма, обученного на предыдущем шаге.
	* Использование: `py linear_regression_sgd_predict.py`
	* Входные данные : 
         * SGD_scaler_{0/90}.npz 
         * SGD_{0/90}_{H|I}.npz
         * <стандартный код> : строка с 11 номерами, соответствующими входным функциям
	* Вывод: <стандартный вывод>: прогнозируемые значения для целевых столбцов. 
	* Примечания: Программа выполняется в бесконечном цикле. На каждой итерации он считывает с консоли строку из 12 числовых параметров, разделенных пробелом. После прочтения он отобразит оценочное значение для каждого целевого столбца. Пустая строка при вводе прервет бесконечный цикл, что приведет к завершению программы.

3. **neural_network_train.py**
	* Цель: Реализация многослойной нейронной сети. Он считывает данные из файлов Excel, а затем обучает нейронную сеть. По завершении он сохраняет состояние нейронной сети во внешних файлах, включая масштабатор, используемый для нормализации данных.
	* Использование: `py neural_network_train.py` 
	* Ввод : 
         * X_bp.xlsx 
         * X_nup.xlsx 
	* Вывод: 
         * NN_scaler_{0/90}_A.pckl 
         * NN_{0/90}_A.pckl 

4. **neural_network_predict.py**
	* Цель: Предсказатель многослойной нейронной сети, обученный на предыдущем шаге.
	* Использование: `py neural_network_predict.py`
	* Входные данные:
         * NN_scaler_{0/90}_A.pckl 
         * NN_{0/90}_A.pckl 
         * <стандартный код> : строка с 12 номерами, соответствующими входным функциям
	* Вывод: <стандартный вывод>: прогнозируемые значения для целевого столбца. 
	* Примечания: Программа выполняется в бесконечном цикле. На каждой итерации он считывает с консоли строку из 12 числовых параметров, разделенных пробелом. После прочтения он отобразит оценочное значение для целевого столбца. Пустая строка при вводе прервет бесконечный цикл, что приведет к завершению программы.

Файлы данных:
-------------

5. **Предоставленные пользователем файлы данных (формат Excel)**:
	* X_bp.xlsx 
   	* X_nup.xlsx 
	* Примечания: Файлы содержат обучающие образцы для алгоритмов обучения, используемых основными программами (см. предыдущий раздел).

6. **Динамически генерируемые файлы основными программами (см. раздел выше)**.
	* SGD_scaler_{0/90}.npz
	* SGD_{0/90}_{H|I}.npz
	* NN_scaler_{0/90}_A.pckl 
	* NN_{0/90}_A.pckl
7. **Результаты поиска по сетке для оптимальных параметров нейронной сети (см. Выше)**
	* Neural-Network-Training.log.txt 

Утилиты
-------

Описание: Различные скрипты, используемые для анализа данных, поиска по сетке и т. Д.

8. **bin\data_analysis.py**
	* Цель: выполняет различный статистический анализ входных данных
	* Использование: `py bin\data_analysis.py`
	* Ввод:
         * X_bp.xlsx 
         * X_nup.xlsx
	* Вывод: <различные графики>

9. **bin\histograms.py**
	* Цель: построение графика распределения входных данных до и после нормализации
	* Использование: `py bin\histograms.py`
	* Ввод:
         * X_bp.xlsx 
         * X_nup.xlsx
	* Вывод: <различные графики>
10. **bin\grid_search.py**
	* Цель: оценивает производительность нейронной сети глубокого обучения по различным параметрам
	* Использование: `py bin\grid_search.py`
	* Ввод:
         * X_bp.xlsx 
         * X_nup.xlsx
	* Вывод:
         * Neural-Network-Training.log.txt 
         * <стандартный вывод>: (совпадает с содержимым файла, упомянутого выше)

Документы
---------
