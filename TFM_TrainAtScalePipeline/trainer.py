from sklearn.linear_model import LinearRegression, Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TFM_TrainAtScalePipeline.encoders import TimeFeaturesEncoder, DistanceTransformer
from TFM_TrainAtScalePipeline.utils import compute_rmse
from TFM_TrainAtScalePipeline.data import get_data, clean_data, df_optimized
from sklearn.model_selection import train_test_split
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
import joblib
from google.cloud import storage
from TFM_TrainAtScalePipeline.params import BUCKET_NAME , STORAGE_LOCATION

from sklearn.preprocessing import FunctionTransformer

class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[GER] [MUC] [moritzbewerunge] TaxiFareModel_783"

    def set_pipeline(self,estimator,transformer):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        reduced_pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('reduced_size', transformer)
        ])
        pipe = Pipeline([
            ('preproc', reduced_pipe),
            ('linear_model', estimator)
        ])
        return pipe

    def run(self,estimator,transformer):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline(estimator,transformer).fit(self.X,self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(f"The root mean squared error is: {rmse}")
        return rmse

    # MLFLOW CLIENT
    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    # Saving the Model
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')

        # Implement here
        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(STORAGE_LOCATION)

        blob.upload_from_filename('model.joblib')
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df_cleaned = clean_data(df)
    # reduce size
    df_reduced = df_optimized(df_cleaned)
    # set X and y
    y = df_reduced["fare_amount"]
    X = df_reduced.drop("fare_amount", axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train

    LinReg = LinearRegression()
    Lass = Lasso()
    estimators = [LinReg, Lass]
    transformer = FunctionTransformer(df_optimized, validate=False)

    for estimator in estimators:
        trainer = Trainer(X_train,y_train)
        trainer.run(estimator,transformer)
        trainer.save_model()
        # evaluate

        evaluated_model = trainer.evaluate(X_val,y_val)

        trainer.mlflow_log_metric("rmse", evaluated_model)
        trainer.mlflow_log_param("model", estimator)
        trainer.mlflow_log_param("student_name", trainer.experiment_name)



    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
