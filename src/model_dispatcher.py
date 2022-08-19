from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier

MODEL = {
    "decisiontree": DecisionTreeClassifier(random_state=0),
    "extratree": ensemble.ExtraTreeClassifier(n_estimators=200, n_jobs=-1, verbose=2)
}