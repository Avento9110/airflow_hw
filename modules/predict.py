import os
import dill
import pandas as pd
import json
from datetime import datetime


path = os.environ.get('PROJECT_PATH', '..')


def predict():
    model_filename = f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'
    with open(model_filename, 'rb') as file:
        model = dill.load(file)
    pred_list = []
    directory = (f'{path}/data/test')
    for testname in os.listdir(directory):
        temp = os.path.join(directory, testname)
        with open(temp) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            pred_list.append([form['id'], y[0]])

    df_predict = pd.DataFrame(pred_list, columns=['car_id','pred'])
    df_predict.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
