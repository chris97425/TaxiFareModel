import pandas as pd
from sklearn.model_selection import train_test_split

def cut_train_csv():
    data_df = pd.read_csv("./raw_data/train.csv",nrows=1000)
    data_df_2 = pd.read_csv("./raw_data/train.csv",nrows=10000)
    data_df.to_csv("./raw_data/train_1k.csv")
    data_df_2.to_csv("./raw_data/train_10k.csv")

def get_data(number_of_line="10k"):
    '''return a dataframe with number of
    line define in number_of_line'''
    df = pd.read_csv(f'./raw_data/train_{number_of_line}.csv')
    return df

def clean_data(df=get_data(), test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def separate_features_predict(df):
    y = df.pop("fare_amount")
    X = df
    return X,y

def split_data(X,y, percent =0.3):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=percent)
    return X_train, X_test, y_train, y_test
