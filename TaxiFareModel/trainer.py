from TaxiFareModel.data import clean_data, separate_features_predict, split_data
from sklearn import set_config; set_config(display='diagram')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans',DistanceTransformer()),
                              ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([('time_trans',TimeFeaturesEncoder('pickup_datetime')),
                              ('oneHotencodeTime', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        self.pipeline = Pipeline([
        ('preproc', preproc_pipe),
        ('linear_model', LinearRegression())])



    def run(self):
        """set and train the pipeline"""
        print(type(self.pipeline))
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    df = clean_data()
    X,y = separate_features_predict(df)
    X_train, X_test, y_train, y_test = split_data(X,y)
    train = Trainer(X=X_train, y=y_train)
    train.set_pipeline()
    train.run()
    result = train.evaluate(X_test, y_test)
    print(result)
