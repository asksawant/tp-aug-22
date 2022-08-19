import pandas as pd
import numpy as np
from sklearn import model_selection

if __name__ == "__main__":

    # Read training data
    df_train = pd.read_csv("input/train.csv")
    print("Shape of the dataset:", df_train.shape)

    # we create a new colum called kfold and fill it with -1
    df_train['kfold'] = -1

    # selecting the target
    y = df_train.failure.values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # fill the new kfold column
    for fold, (train_indicies,valid_indicies) in enumerate(kf.split(X=df_train,y=y)):
        print(len(train_indicies),len(valid_indicies))
        df_train.loc[valid_indicies,'kfold'] = fold

    # save the new csv with kfold column
    df_train.to_csv("input/train_fold.csv",index=False)