'''make and learn model'''
import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop


def make_dataset():
    '''make dataset'''
    # load learn_data
    test_data = pd.read_csv('data/test_data.csv', index_col=0)
    test_data = test_data.reset_index(drop=True)
    for i in test_data.columns:
        print(i)
    print(test_data)

    drop_index = test_data.index[(test_data['着順'].str.contains('中')) |
                                 (test_data['着順'].str.contains('除')) |
                                 (test_data['着順'].str.contains('取')) |
                                 (test_data['着順'].str.contains('降')) |
                                 (test_data['着順'].str.contains('失')) |
                                 (test_data['単勝'].str.contains('---'))]
    test_data = test_data.drop(drop_index)
    test_data = test_data.reset_index(drop=True)
    print(test_data)

    # 型変換
    labels = ['index', 'レース名', '馬名']
    for label in test_data.columns:
        if label in labels:
            continue
        test_data[label] = test_data[label].astype(float)
    # for label in test_data.columns:
    #     print(f'{label}:{test_data[label].dtype}')
    test_data.to_csv('bbb.csv')

    # raceごとに区切る
    race_top_index = test_data.query('num == 0').index
    print(race_top_index)
    races = []
    for i in range(len(race_top_index)-1):
        start = race_top_index[i]
        end = race_top_index[i+1]
        race = test_data.iloc[start:end]
        race.reset_index(drop=True, inplace=True)
        races.append(race)
        # print(i)
        # print(f'start:{start}')
        # print(f'end:{end}')
        # print(races[i])
    # print(races)

    # x and y(ans)
    # drops = ['レース名', '馬名', 'タイム指数', '上り', '着順', '馬id']
    # x = test_data.drop(drops, axis=1)
    # y = test_data['タイム指数']
    return races


def test(races):
    model = keras.models.load_model('./model/mymodel.h5')

    # results(レース名, 馬名(一位), 収支, odds, back,)
    results = []

    for race in races[3:]:
        print(race)
        result = [race['レース名'][0]]
        # x and y(ans)
        drops = ['num', 'レース名', '馬名', 'タイム指数', '上り', '着順', '馬id']
        x = race.drop(drops, axis=1)
        y = race['タイム指数']
        predicts = model.predict(x)
        predicts = pd.Series(predicts.T[0])
        for i, predict in enumerate(predicts):
            print('ans:{:.1f}, predict:{}, diff:{:.1f}'.format(predict, y.iloc[i], abs(predict-y.iloc[i])))
        race = pd.concat([race, predicts], axis=1, join='inner')
        race.sort_values('着順')
        race.to_csv('aaa.csv')

        break


if __name__ == '__main__':
    races = make_dataset()
    test(races)
