import pandas as pd
import numpy as np
from sklearn import model_selection

if __name__ == "__main__":

    # Read the training data
    df_train = pd.read_csv("input/train.csv")
    print("Shape of the dataset: ",df_train.shape)

    # Creating the a new column called kfold and assigning the value -1
    df_train['kfold'] = -1
    print("Shape of the dataset after adding kfold col: ",df_train.shape)
    # Shuffling the data
    # df = df.sample(frac=1).reset_index(drop=True)

    # Selecting the target variables
    y = df_train.failure.values

    # initiate the kfold class from model_selection module
    # as this is imbalance dataset
    # using StratifiedKfold

    kf = model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # fill the new kfold column
    for fold, (train_indicies,valid_indicies) in enumerate(kf.split(X=df_train,y=y)):
        print(len(train_indicies),len(valid_indicies))
        df_train.loc[valid_indicies,'kfold'] = fold
    
    # save the new csv with kfold column
    df_train.to_csv("input/train_folds.csv", index = False)
