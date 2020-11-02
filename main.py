import pandas as pd
import algorithm as alg

titanic = pd.read_csv("titanic.csv")
del titanic["Name"]


def run_n(dataset, n=10):
    total_knn = 0
    total_nb = 0
    total_logreg = 0
    for i in range(n):
        total_knn += alg.k_nearest_neighbour(dataset)
        total_nb += alg.naiv_bayes(dataset)
        total_logreg += alg.log_reg(dataset)
    score_knn = total_knn/n
    score_nb = total_nb/n
    score_logreg = total_logreg/n

    print()
    print("After running",n,"times, the average score is:")
    print()
    print("Avg. score of k-nearest neighbour: ", score_knn)
    print("Avg. score of naiv bayes:          ", score_nb)
    print("Avg. score of logistic regression: ", score_logreg)


run_n(titanic)
