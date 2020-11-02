import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

titanic = pd.read_csv("titanic.csv")
del titanic["Name"]


def replace_gender(dataframe, rep_male="male", rep_female="female"):
    dataframe = dataframe.replace(to_replace=rep_male, value=1)
    dataframe = dataframe.replace(to_replace=rep_female, value=0)
    return dataframe


def scale_df(dataframe):
    scaler = MinMaxScaler()
    newdataframe = scaler.fit_transform(dataframe)
    newdataframe = pd.DataFrame(newdataframe, columns=dataframe.columns)
    return newdataframe


def categories_age_fare(dataframe, age="Age", fare="Fare"):
    dataframe = replace_gender(dataframe)
    age_25p = np.percentile(dataframe[age], 25)
    age_50p = np.percentile(dataframe[age], 50)
    age_75p = np.percentile(dataframe[age], 75)
    age_df = dataframe[age]
    for n in age_df.values:
        if n >= age_75p:
            age_df = age_df.replace(to_replace=n, value=0)
        elif n >= age_50p:
            age_df = age_df.replace(to_replace=n, value=1)
        elif n >= age_25p:
            age_df = age_df.replace(to_replace=n, value=2)
        elif n < age_25p:
            age_df = age_df.replace(to_replace=n, value=3)
    dataframe[age] = age_df

    fare_25p = np.percentile(dataframe[fare], 25)
    fare_50p = np.percentile(dataframe[fare], 50)
    fare_75p = np.percentile(dataframe[fare], 75)
    fare_df = dataframe[fare]
    for n in fare_df.values:
        if n >= fare_75p:
            fare_df = fare_df.replace(to_replace=n, value=0)
        elif n >= fare_50p:
            fare_df = fare_df.replace(to_replace=n, value=1)
        elif n >= fare_25p:
            fare_df = fare_df.replace(to_replace=n, value=2)
        elif n < fare_25p:
            fare_df = fare_df.replace(to_replace=n, value=3)
    dataframe[fare] = fare_df
    return dataframe


def tts(dataset, y="Survived"):
    x = dataset.drop(columns=[y])
    y = dataset[y].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test
