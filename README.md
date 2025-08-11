# Binary-Classification-with-a-Bank-Dataset
The goal is to predict whether a client will subscribe to a bank term deposit.

```mermaid
stateDiagram-v2
    %% Data Loading States
    state "Data Loading" as dl {
        r_train_d: Read Training Data
        r_test_d: Read Test Data
        gf: Extract Features
        gt: Extract Target
    }

    %% Data Processing States
    state "Data Processing" as dp {
        ppd1: Preprocess Training Data
        ppd2: Preprocess Test Data
        sd: Train/Val Split
    }

    %% Model States
    state "Model Pipeline" as mp {
        t_ml_m: Train Model
        mp1: Training Predictions
        mp2: Test Predictions
        em: Evaluate Model
    }

    %% Flow
    r_train_d --> gf
    r_train_d --> gt
    gf --> ppd1
    ppd1 --> sd
    gt --> sd
    sd --> t_ml_m
    t_ml_m --> mp1
    mp1 --> em
    r_test_d --> ppd2
    t_ml_m --> mp2
    ppd2 --> mp2
    mp2 --> em
```

```bibtex
@misc{playground-series-s5e8,
    author = {Walter Reade and Elizabeth Park},
    title = {Binary Classification with a Bank Dataset},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e8}},
    note = {Kaggle}
}
```
