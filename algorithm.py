from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import modifier as mod

titanic = pd.read_csv("titanic.csv")
del titanic["Name"]


def k_nearest_neighbour(dataset):
    dataset = mod.replace_gender(dataset)
    dataset = mod.scale_df(dataset)
    other_train, other_test, survived_train, survived_test = mod.tts(dataset)
    knn = KNeighborsClassifier()
    knn.fit(other_train, survived_train)
    # survive_pred = knn.predict(other_test)
    return knn.score(other_test, survived_test)


def naiv_bayes(dataset):
    dataset = mod.replace_gender(dataset)
    dataset = mod.categories_age_fare(dataset)
    other_train, other_test, survived_train, survived_test = mod.tts(dataset)
    nb = MultinomialNB()
    nb.fit(other_train, survived_train)
    # survive_pred = nb.predict(other_test)
    return nb.score(other_test, survived_test)


def log_reg(dataset):
    dataset = mod.replace_gender(dataset)
    dataset = mod.scale_df(dataset)
    other_train, other_test, survived_train, survived_test = mod.tts(dataset)
    logreg = LogisticRegression()
    logreg.fit(other_train, survived_train)
    # survive_pred = logreg.predict(other_test)
    return logreg.score(other_test, survived_test)
