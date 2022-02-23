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
    learn_data = pd.read_csv('data/allrace.csv', index_col=0)
    learn_data = learn_data.reset_index()

    drop_index = learn_data.index[(learn_data['着順'].str.contains('中')) |
                                  (learn_data['着順'].str.contains('除')) |
                                  (learn_data['着順'].str.contains('取')) |
                                  (learn_data['着順'].str.contains('降')) |
                                  (learn_data['着順'].str.contains('失')) |
                                  (learn_data['単勝'].str.contains('---'))]
    learn_data = learn_data.drop(drop_index)

    # learn_data = learn_data.sample(frac=1, random_state=0)

    labels = ['index', 'レース名', '馬名']
    for label in learn_data.columns:
        if label in labels:
            continue
        # print(label)
        learn_data[label] = learn_data[label].astype(float)
    # for label in learn_data.columns:
    #     print(f'{label}:{learn_data[label].dtype}')

    learn_data = learn_data.reset_index()

    # x and y(ans)
    drops = ['レース名', '馬名', 'タイム指数', '上り', '着順', '馬id']
    x = learn_data.drop(drops, axis=1)
    y = learn_data['タイム指数']

    # 振り分け
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


class Model:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.epochs = 10
        self.history = None
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_train = X_train
        self.Y_train = Y_train
        self.model = self.build_model()

    def build_model(self):
        '''make and compile model'''
        input_shape = (len(self.X_train.columns), 0)
        input_shape = 38
        # model
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(input_shape, )))
        model.add(BatchNormalization(epsilon=0.001))
        # model.add(keras.layers.Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization(epsilon=0.001))
        # model.add(keras.layers.Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization(epsilon=0.001))
        # model.add(keras.layers.Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization(epsilon=0.001))
        # model.add(keras.layers.Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization(epsilon=0.001))
        # model.add(keras.layers.Dropout(0.3)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization(epsilon=0.001))
        # model.add(keras.layers.Dropout(0.3))
        model.add(Dense(1))
        model.summary()

        optimizer = Adam(learning_rate=0.03)
        # optimizer = RMSprop(learning_rate=0.03)
        # mse = MeanSquaredError()
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['accuracy']
                      # metrics=['mae']
                      )
        return model

    def make_callbacks(self):
        early_stopping = EarlyStopping(
                            monitor='val_loss',  # 監視対象
                            # patience=100,  # 最低ループ数
                            patience=20,
                            verbose=0,  # 保存時の出力にコメント
                            mode='auto'  # 収束判定
                         )
        learning_rates = np.linspace(0.01, 0.0001, self.epochs)
        learning_rateScheduler = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))

        callbacks = [early_stopping, learning_rateScheduler]
        return callbacks

    def fit(self):
        callbacks = self.make_callbacks()
        params = {
            'x': X_train,
            'y': Y_train,
            'batch_size': 100,
            'epochs': self.epochs,
            'verbose': 1,
            'validation_split': 0.2,
            # 'validation_data': (X_test, Y_test),
            'callbacks': callbacks,
        }
        self.history = self.model.fit(**params)
        self.model.save("./model/mymodel.h5")

    def display_result(self):
        '''display result with plt'''

        history = self.history
        # print(history.history)

        # plot
        plt.title('Model loss')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('Loss')
        # plt.title('Model accuracy')
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.ylabel('Accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.ylim(100, 500)
        plt.show()

    def test(self):
        X_test = self.X_test
        Y_test = self.Y_test

        # roopのためにindex降り直し
        X_test.reset_index(inplace=True, drop=True)
        Y_test.reset_index(inplace=True, drop=True)

        results = self.model.predict(X_test, batch_size=1, verbose=1)
        dif_rate_sum = 0
        count = 0
        for i, result in enumerate(results):
            if Y_test[i]:
                count += 1
                dif = abs(Y_test[i]-result)
                dif_rate_sum += dif/Y_test[i]*100
                accuracy = dif_rate_sum/count
                print(f'{i+1}: {result},{Y_test[i]}:{dif}({accuracy}%)')
            else:
                print(f'{i+1}:')


if __name__ == '__main__':
    # make dataset
    X_train, X_test, Y_train, Y_test = make_dataset()

    # make&compile model
    model = Model(X_train, X_test, Y_train, Y_test)
    model.fit()
    model.display_result()
    model.test()
