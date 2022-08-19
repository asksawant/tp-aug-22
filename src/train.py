import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# TRAINING_DATA = os.environ.get("TRAINING_DATA")
# FOLD = os.environ.get("FOLD")

TRAINING_DATA = input/train_fold.csv
FOLD = 0

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kold == FOLD]

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

    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])

    # Preprocessing of training data, fit model
    clf.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = clf.predict_proba(X_valid)[:,1]
    print(preds)





