'''make and learn model'''
from keras.models import load_model

import pandas as pd

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

    # raceごとに区切る
    race_top_index = test_data.query('num == 0').index
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


def backtest(races):
    model = load_model('./model/mymodel.h5')
    # results(レース名, ○or×, 馬名(一位), 収支, odds, back,)
    results = []
    race_num = 0
    error_race_num = 0
    odds_sum = 0
    back = 0

    for race in races:
        try:
            # 型変換
            labels = ['index', 'レース名', '馬名']
            for label in race.columns:
                if label in labels:
                    continue
                race[label] = race[label].astype(float)
        except ValueError:
            # エラー値を含むレースは除外する
            error_race_num += 1
            continue
        print('\r', f'race_num:{race_num}/{len(races)}, error_race_num:{error_race_num}/{len(races)}', end='')

        # 予測&dataframe作成
        # print(race)
        race_num += 1
        result = [race['レース名'][0]]
        # x and y(ans)
        drops = ['num', 'レース名', '馬名', 'タイム指数', '上り', '着順', '馬id']
        x = race.drop(drops, axis=1)
        predicts = pd.Series(model.predict(x).T[0]).rename('predict')
        # y = race['タイム指数']
        # for i, predict in enumerate(predicts):
        #     print('ans:{:.1f}, predict:{}, diff:{:.1f}'.format(predict, y.iloc[i], abs(predict-y.iloc[i])))
        race = pd.concat([race, predicts], axis=1, join='inner')
        race = race.sort_values(by=['predict'], ascending=False)
        race.reset_index(drop=True, inplace=True)

        # 判定
        horse_name = race.iloc[0, 5]
        odds = race.iloc[0, 10]
        if int(race.iloc[0, 2]) == 1:
            result.append('○')
            result.append(horse_name)
            odds_sum += odds
            back += odds-1
            result.append(odds_sum)
            result.append(str(odds))
            result.append(back)
        else:
            result.append('×')
            result.append(horse_name)
            odds_sum += -1
            back += -1
            result.append(odds_sum)
            result.append(str('-'))
            result.append(back)
        result.append(len(race))

        results.append(result)
    print(f'race_num:{race_num}')
    print(f'error_race_num:{error_race_num}')

    results = pd.DataFrame(results)
    columns = ['レース名', '○|×', '馬名', '収支', 'オッズ', '利益', '頭数']
    results.columns = columns
    results.to_csv('result.csv')


if __name__ == '__main__':
    Races = make_dataset()
    backtest(Races)
