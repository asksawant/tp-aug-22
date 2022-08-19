import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

TRAINING_DATA = "input/train_fold.csv"
TEST_DATA = "input/test.csv"

def run(fold):

    df= pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    predictions = None

    features = [f for f in df.columns if f not in ('id','kfold','failure')]

    df_test = df_test[features]

    # final_predictions = []

    # for fold in range(5):

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    y_train = train_df.failure.values
    y_valid = valid_df.failure.values

    train_df = train_df.drop(['id','failure','kfold'], axis=1)
    valid_df = valid_df.drop(['id','failure','kfold'], axis=1)

    valid_df = valid_df[train_df.columns]

    # Select numerical columns
    numerical_cols = [cname for cname in train_df.columns if train_df[cname].dtype in ['int64', 'float64']]

    # Select categorical columns
    categorical_cols = [cname for cname in train_df.columns if train_df[cname].dtype == "object"]

    # keep selected columns only
    my_cols = numerical_cols + categorical_cols
    X_train = train_df[my_cols].copy()
    X_valid = valid_df[my_cols].copy()
    X_test = df_test[my_cols].copy()

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='mean')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy ='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # data is ready to train
    # Defining the model
    model = DecisionTreeClassifier(random_state=0)
    # model = model_dispatcher.MODEL(MODEL)

    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])

    # Preprocessing of training data, fit model
    clf.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = clf.predict_proba(X_valid)[:,1]
    print(preds)
    score = metrics.roc_auc_score(y_valid,preds)

    # print auc
    print(f"Fold = {fold}, AUC = {score}")

    # getting predictions for test
    test_preds = clf.predict_proba(X_test)[:,1]
    print(test_preds)

    return test_preds


if __name__ == "__main__":

    sample_submission = pd.read_csv("input/sample_submission.csv")
    for fold in range(5):
        run(fold)
        if fold == 0:
            predictions = test_preds
        else:
            predictions += test_preds
    predictions /= 5
    final_predict = np.column_stack(predictions)
    print("Shape of prediction: ",final_predict.shape)
    print("Shape of sample_submssion file:",sample_submission.shape)
    # sample_submission.failure = final_predict


