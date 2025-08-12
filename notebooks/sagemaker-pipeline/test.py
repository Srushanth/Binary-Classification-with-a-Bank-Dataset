"""
Enhanced SageMaker Pipeline with proper parameterization and input/output patterns
similar to Kubeflow's Input and Output definitions
"""

import sagemaker
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.function_step import step
from sagemaker.workflow.step_outputs import get_step
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat,
)
from typing import Tuple


# Define Pipeline Parameters (similar to Kubeflow Inputs)
train_data_path = ParameterString(
    name="TrainDataPath",
    default_value="s3://srushanth-baride/binary-classification-with-a-bank-dataset/train.csv",
)

test_data_path = ParameterString(
    name="TestDataPath",
    default_value="s3://srushanth-baride/binary-classification-with-a-bank-dataset/test.csv",
)

instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

instance_count = ParameterInteger(name="InstanceCount", default_value=1)

train_test_split_ratio = ParameterFloat(name="TrainTestSplitRatio", default_value=0.8)

random_state = ParameterInteger(name="RandomState", default_value=42)


@step(
    name="IngestTrainingData",
    display_name="Ingest Training Data",
    instance_type=instance_type,
    instance_count=instance_count,
)
def ingest_train_data(s3_path: str) -> pd.DataFrame:
    """
    Ingest training data from S3 path

    Args:
        s3_path (str): S3 path to training data

    Returns:
        pd.DataFrame: Training dataframe
    """
    train_df = pd.read_csv(s3_path)
    print(f"Ingested training data with shape: {train_df.shape}")
    return train_df


@step(
    name="IngestTestData",
    display_name="Ingest Test Data",
    instance_type=instance_type,
    instance_count=instance_count,
)
def ingest_test_data(s3_path: str) -> pd.DataFrame:
    """
    Ingest test data from S3 path

    Args:
        s3_path (str): S3 path to test data

    Returns:
        pd.DataFrame: Test dataframe
    """
    test_df = pd.read_csv(s3_path)
    print(f"Ingested test data with shape: {test_df.shape}")
    return test_df


@step(
    name="ExtractFeatures",
    display_name="Extract Feature Column",
    instance_type=instance_type,
    instance_count=instance_count,
)
def extract_target_column(train_df: pd.DataFrame) -> pd.Series:
    """
    Extract target column from training data

    Args:
        train_df (pd.DataFrame): Training dataframe

    Returns:
        pd.Series: Target variable series
    """
    target_column = train_df["y"]
    print(f"Extracted target column with {len(target_column)} samples")
    return target_column


@step(
    name="PreprocessTrainingData",
    display_name="Preprocess Training Data",
    instance_type=instance_type,
    instance_count=instance_count,
)
def preprocess_train_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess training data with label encoding

    Args:
        df (pd.DataFrame): Raw training dataframe

    Returns:
        tuple: Processed features (X) and target (y)
    """
    # Create a copy to avoid modifying original
    processed_df = df.copy()

    # Label encoding for categorical columns
    categorical_columns = [
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]

    for column_name in categorical_columns:
        if column_name in processed_df.columns:
            le = LabelEncoder()
            processed_df[column_name] = le.fit_transform(processed_df[column_name])

    # Split features and target
    y = processed_df["y"]
    X = processed_df.drop(columns=["id", "y"], errors="ignore")

    print(
        f"Preprocessing completed - Features shape: {X.shape}, Target shape: {y.shape}"
    )
    return X, y


@step(
    name="PreprocessTestData",
    display_name="Preprocess Test Data",
    instance_type=instance_type,
    instance_count=instance_count,
)
def preprocess_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess test data with same transformations as training

    Args:
        df (pd.DataFrame): Raw test dataframe

    Returns:
        pd.DataFrame: Processed test features
    """
    # Create a copy to avoid modifying original
    processed_df = df.copy()

    # Label encoding for categorical columns
    categorical_columns = [
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]

    for column_name in categorical_columns:
        if column_name in processed_df.columns:
            le = LabelEncoder()
            processed_df[column_name] = le.fit_transform(processed_df[column_name])

    # Remove ID column if present
    if "id" in processed_df.columns:
        processed_df = processed_df.drop(columns=["id"])

    print(f"Test data preprocessing completed - Shape: {processed_df.shape}")
    return processed_df


@step(
    name="SplitTrainingData",
    display_name="Split Training Data",
    instance_type=instance_type,
    instance_count=instance_count,
)
def split_training_data(
    X: pd.DataFrame, y: pd.Series, train_size: float, random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and validation sets

    Args:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target series
        train_size (float): Proportion for training
        random_seed (int): Random state for reproducibility

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_size, random_state=random_seed
    )

    print(f"Data split completed:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape} samples")

    return X_train, X_val, y_train, y_val


@step(
    name="TrainModel",
    display_name="Train LightGBM Model",
    instance_type=instance_type,
    instance_count=instance_count,
)
def train_lightgbm_model(
    X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series
) -> lgb.LGBMClassifier:
    """
    Train LightGBM classifier model

    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        y_train (pd.Series): Training target
        y_val (pd.Series): Validation target

    Returns:
        lgb.LGBMClassifier: Trained model
    """
    # Initialize model with some basic parameters
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=0,
    )

    # Train with evaluation set
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
    )

    print("Model training completed successfully")
    return model


@step(
    name="GeneratePredictions",
    display_name="Generate Predictions",
    instance_type=instance_type,
    instance_count=instance_count,
)
def generate_predictions(
    model: lgb.LGBMClassifier, test_data: pd.DataFrame
) -> np.ndarray:
    """
    Generate predictions on test data

    Args:
        model (lgb.LGBMClassifier): Trained model
        test_data (pd.DataFrame): Test features

    Returns:
        np.ndarray: Prediction probabilities
    """
    # Generate probability predictions for positive class
    predictions = model.predict_proba(test_data)[:, 1]

    print(f"Generated predictions for {len(predictions)} samples")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    return predictions


def create_pipeline():
    """
    Create and configure the SageMaker Pipeline

    Returns:
        Pipeline: Configured pipeline instance
    """
    # Step 1: Ingest data
    step_ingest_train = ingest_train_data(train_data_path)
    step_ingest_test = ingest_test_data(test_data_path)

    # Step 2: Preprocess data
    step_preprocess_train = preprocess_train_data(step_ingest_train)
    step_preprocess_test = preprocess_test_data(step_ingest_test)

    # Step 3: Split training data
    step_split_data = split_training_data(
        X=step_preprocess_train[0],  # Features from preprocessing
        y=step_preprocess_train[10],  # Target from preprocessing
        train_size=train_test_split_ratio,
        random_seed=random_state,
    )

    # Step 4: Train model
    step_train_model = train_lightgbm_model(
        X_train=step_split_data[0],
        X_val=step_split_data[10],
        y_train=step_split_data[20],
        y_val=step_split_data[21],
    )

    # Step 5: Generate predictions
    step_predictions = generate_predictions(
        model=step_train_model, test_data=step_preprocess_test
    )

    # Create pipeline with parameters and steps
    pipeline = Pipeline(
        name="EnhancedBankMarketingPipeline",
        parameters=[
            train_data_path,
            test_data_path,
            instance_type,
            instance_count,
            train_test_split_ratio,
            random_state,
        ],
        steps=[
            step_ingest_train,
            step_ingest_test,
            step_preprocess_train,
            step_preprocess_test,
            step_split_data,
            step_train_model,
            step_predictions,
        ],
        sagemaker_session=sagemaker.Session(),
    )

    return pipeline


if __name__ == "__main__":
    # Get execution role
    role_arn = sagemaker.get_execution_role()

    # Create pipeline
    pipeline = create_pipeline()

    # Upsert pipeline (create or update)
    pipeline.upsert(role_arn=role_arn)
    print(f"Pipeline '{pipeline.name}' created/updated successfully")

    # Start pipeline execution with default parameters
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")

    # Optional: Start with custom parameters
    # execution_custom = pipeline.start(
    #     parameters={
    #         "InstanceType": "ml.m5.large",
    #         "TrainTestSplitRatio": 0.75,
    #         "RandomState": 123
    #     }
    # )
