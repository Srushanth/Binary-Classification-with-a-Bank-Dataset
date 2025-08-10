"""_summary_
"""

import sagemaker
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.function_step import step
from sagemaker.workflow.step_outputs import get_step

@step(instance_type="ml.m5.xlarge", instance_count=1)
def ingest_data(s3_path: str) -> pd.DataFrame:
    """_summary_

    Args:
        s3_path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    train_df = pd.read_csv(s3_path)
    return train_df

@step(instance_type="ml.m5.xlarge", instance_count=1)
def get_feature_column(train_df: pd.DataFrame) -> pd.Series:
    """_summary_

    Args:
        train_df (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    return train_df["y"]

@step(instance_type="ml.m5.xlarge", instance_count=1)
def preprocess_data(df: pd.DataFrame) -> tuple:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        tuple: _description_
    """
    # Label encoding
    object_labels = [
        "job", 
        "marital", 
        "education", 
        "default", 
        "balance", 
        "housing", 
        "loan", 
        "contact", 
        "month", 
        "poutcome"
    ]

    for column_name in object_labels:
        le = LabelEncoder()
        df[column_name] = le.fit_transform(df[column_name])

    # Split features and target
    y = df["y"]
    X = df.drop(columns=["id", "y"])

    return X, y

@step(instance_type="ml.m5.xlarge", instance_count=1)
def split_data(X: pd.DataFrame, y: pd.DataFrame) -> tuple:
    """_summary_

    Args:
        X (pd.DataFrame): _description_
        y (pd.DataFrame): _description_

    Returns:
        tuple: _description_
    """
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    return x_train, x_test, y_train, y_test

@step(instance_type="ml.m5.xlarge", instance_count=1)
def train_model(
    x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame
) -> lgb.LGBMClassifier:
    """_summary_

    Args:
        x_train (pd.DataFrame): _description_
        x_test (pd.DataFrame): _description_
        y_train (pd.DataFrame): _description_
        y_test (pd.DataFrame): _description_

    Returns:
        lgb.LGBMClassifier: _description_
    """
    model = lgb.LGBMClassifier()
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
    
    return model

@step(instance_type="ml.m5.xlarge", instance_count=1)
def make_predictions(model: lgb.LGBMClassifier, test_data: pd.DataFrame) -> np.ndarray:
    """_summary_

    Args:
        model (lgb.LGBMClassifier): _description_
        test_data (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """
    test_data = test_data.drop(columns=["id"])
    return model.predict_proba(test_data)[:, -1]

def create_pipeline():
    # Define the pipeline steps
    train_data_path = "s3://srushanth-baride/binary-classification-with-a-bank-dataset/train.csv"
    test_data_path = "s3://srushanth-baride/binary-classification-with-a-bank-dataset/test.csv"

    # Ingest data
    step_ingest_train_data = ingest_data(train_data_path)

    # Preprocess data
    step_preprocess = preprocess_data(step_ingest_train_data)

    # Split data
    step_split = split_data(
        step_preprocess[0],
        step_preprocess[1]
    )
    
    # Train model
    step_train = train_model(
        step_split[0],
        step_split[1],
        step_split[2],
        step_split[3]
    )

    # Make predictions
    test_df = pd.read_csv(test_data_path)
    step_predict = make_predictions(
        step_train,
        test_df
    )

    # If you really need explicit dependencies, define them like this:
    step_preprocess_instance = get_step(step_preprocess)
    step_split_instance = get_step(step_split)
    step_train_instance = get_step(step_train)
    step_predict_instance = get_step(step_predict)

    # Add dependencies if needed
    step_split_instance.add_depends_on([step_preprocess_instance])
    step_train_instance.add_depends_on([step_split_instance])
    step_predict_instance.add_depends_on([step_train_instance])

    # Create and return pipeline
    pipeline = Pipeline(
        name="BankMarketingPipeline",
        steps=[step_ingest_train_data, step_preprocess, step_split, step_train, step_predict],
        sagemaker_session=sagemaker.Session()
    )
    
    return pipeline

if __name__ == "__main__":
    role_arn = sagemaker.get_execution_role()

    # Execute pipeline
    pipeline = create_pipeline()
    pipeline.upsert(role_arn=role_arn)
    execution = pipeline.start()
