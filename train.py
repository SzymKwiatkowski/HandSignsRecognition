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
    # set run flags
    testing = False
    save_model = False
    
    # load preprocessed training data
    data = pickle.load(open('data/data.pickle', 'rb'))
    x = data['data']
    y = data['labels']
    
    # Convert to numpy array
    x = x.to_numpy()
    
    # Set filename for model to get save to or read from
    filename = 'models/finalized_model.sav'

    # Split data evenly thorought with 8 to 2 proportion
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
    
    if testing:
        # Set polynominal SVC and it's properties
        clf2 = SVC(kernel='poly',probability=True, degree=4, C=12.0)
        
        # Neural network with custom params
        clf1 = MLPClassifier(solver='adam', activation='tanh', learning_rate='adaptive', alpha=1e-5, random_state=42, 
                             shuffle=True, batch_size=16, max_iter=1000, learning_rate_init=0.0008, power_t=0.7)

        # Set VotingClassifier to ensamble both classifiers above
        eclf = VotingClassifier(estimators=[("mlpc", clf1), ('svc', clf2)],
                                voting='soft',
                                weights=[2, 2])

        # Train and predict
        eclf.fit(X_train, y_train)
        y_predict = eclf.predict(X_test)

        # Display accuracy along with score and confusion matrix
        print(accuracy_score(y_test, y_predict))
        print(precision_score(y_test, y_predict, average=None))
        cm = confusion_matrix(y_test, y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=letter_dict.keys())
        disp.plot()
        plt.show()
        
        # Save data if flags are set and file does not exist
        if save_model and not os.path.isfile(filename):
            pickle.dump(eclf, open(filename, 'wb'))
    
    else:
        # Load existing model and run tests on test data
        model = pickle.load(open(filename, 'rb'))
        
        y_predict = model.predict(X_test)
        print(accuracy_score(y_test, y_predict))
        print(precision_score(y_test, y_predict, average=None))
        cm = confusion_matrix(y_test, y_predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=letter_dict.keys())
        disp.plot()
        plt.show()
if __name__ == "__main__":
    main()