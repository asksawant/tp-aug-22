import pandas as pd
import numpy as np
from sklearn import model_selection

if __name__ == "__main__":

    df_train = pd.read_csv("input/train.csv")
    print("Shape of the dataset:", df_train.shape)

    df_train['kfold'] = -1

    y = df_train.failure.values

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_indicies,valid_indicies) in enumerate(kf.split(X=df_train,y=y)):
        print(len(train_indicies),len(valid_indicies))
        df_train.loc[valid_indicies,'kfold'] = fold

    df_train.to_csv("input/train_fold.csv",index=False)