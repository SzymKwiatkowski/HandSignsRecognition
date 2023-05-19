import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os 

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



def main():
    data = pickle.load(open('data/data.pickle', 'rb'))
    # print(data)
    x = data['data']
    y = data['labels']
    x = x.to_numpy()
    filename = 'models/finalized_model.sav'
    testing = True
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
    
    # X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=42, stratify=y_train)
    if not os.path.isfile(filename) and not testing:
        clf2 = SVC(kernel='poly',probability=True, degree=4, C=12.0)
        clf1 = MLPClassifier(solver='adam', activation='tanh', learning_rate='adaptive', alpha=1e-5, random_state=42, 
                             shuffle=True, batch_size=16, max_iter=1000, learning_rate_init=0.0008, power_t=0.7)

        eclf = VotingClassifier(estimators=[("mlpc", clf1), ('svc', clf2)],
                                voting='soft',
                                weights=[2, 2])

        eclf.fit(X_train, y_train)
        y_predict = eclf.predict(X_test)

        print(accuracy_score(y_test, y_predict))
        print(precision_score(y_test, y_predict, average=None))
        cm = confusion_matrix(y_test, y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=letter_dict.keys())
        disp.plot()
        plt.show()
        if not testing:
            pickle.dump(eclf, open(filename, 'wb'))
    
if __name__ == "__main__":
    main()