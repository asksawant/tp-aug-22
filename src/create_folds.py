import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    # Read the training data
    df = pd.read_csv("input/train.csv")

    # Creating the a new column called kfold and assigning the value -1
    df["kfold"] = -1

    # Shuffling the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Selecting the target variables
    y = df.failure.values

    # initiate the kfold class from model_selection module
    # as this is imbalance dataset
    # using StratifiedKfold

    kf = model_selection.StratifiedGroupKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_,v_) in enumerate(kf.split(X = df, y = y)):
        print(len(t_),len(v_))
        df.loc[v_, 'kfold'] = f
    
    # save the new csv with kfold column
    df.to_csv("input/train_folds.csv", index = False)
    