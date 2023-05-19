import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.inspection import DecisionBoundaryDisplay

hand_dict = {
    'Right': 0,
    'Left': 1
}

letter_dict = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'k': 9,
    'l': 10,
    'm': 11,
    'n': 12,
    'o': 13,
    'p': 14,
    'q': 15,
    'r': 16,
    's': 17,
    't': 18,
    'u': 19,
    'v': 20,
    'w': 21,
    'x': 22,
    'y': 23,
}

def plot_decision_boundary(X, y, classifier, title: str):
    plt.figure(figsize=(10, 8))
    X1, X2 = np.meshgrid(
        np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
        np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(
        X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.5, cmap=ListedColormap(('red', 'green', 'blue'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
      plt.scatter(
          X[y == j, 0], X[y == j, 1], color=ListedColormap(('red', 'green', 'blue'))(i), label=j
      )
    plt.title(title)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend()

def main():
    data = pickle.load(open('data/data.pickle', 'rb'))
    x = data['data']
    y = data['labels']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
    # clf4 = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    clf1 = MLPClassifier()
    clf2 = RandomForestClassifier()
    # clf3 = SVC(gamma=0.1, kernel="rbf", probability=True)
    # clf3 = ExtraTreesClassifier()
    
    eclf = VotingClassifier(
    estimators=[("mlp", clf1), ("rf", clf2)],
    voting="soft",
    weights=[2, 2],
)
    
    eclf.fit(X_train, y_train)
    y_predict = eclf.predict(X_test)
    
    print(accuracy_score(y_test, y_predict))
    print(precision_score(y_test, y_predict, average=None))
    # X = x.data[:]
    # plot_decision_boundary(X, y, clf1, 'MLPClass')
    plt.show()
    
    return 0

if __name__ == "__main__":
    main()