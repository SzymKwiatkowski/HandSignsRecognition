import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import os 
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

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

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    # set run flags
    testing = True
    save_model = False
    monitor_network = True
    
    # load preprocessed training data
    data = pickle.load(open('data/data_3.pickle', 'rb'))
    x = data['data']
    y = data['labels']
    
    # Convert to numpy array
    x = x.to_numpy()
    
    # Set filename for model to get save to or read from
    filename = 'models/model.pkl'

    # Split data evenly thorought with 8 to 2 proportion
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
    
    # train params:
    # SVC:
    kernel = 'poly'
    probability = True
    degree = 5
    C = 12.0
    svc_weight = 3.15
    
    # MLP
    solver = 'adam'
    activation = 'relu'
    learning_rate = 'adaptive'
    alpha = 1e-7
    batch_size = 8
    learning_rate_initialize = 0.0015
    power_t = 0.57
    mlp_weight = 3.05
    
    # QDA
    tolerance = 1e-6 * 2
    qda_weight = 3.3
    
    if testing:
        with mlflow.start_run():
            # Set polynominal SVC and it's properties
            clf2 = SVC(kernel=kernel, probability=probability, degree=degree, C=C)

            # Neural network with custom params
            clf1 = MLPClassifier(solver=solver, activation=activation, learning_rate=learning_rate, alpha=alpha, random_state=42, 
                                 shuffle=True, batch_size=batch_size, max_iter=800, warm_start=True, verbose=monitor_network, 
                                 learning_rate_init=learning_rate_initialize, power_t=power_t)

            # Quadratic Classifier with low tolerance
            clf3 = QuadraticDiscriminantAnalysis(tol=tolerance)

            # Set VotingClassifier to ensamble both classifiers above
            eclf = VotingClassifier(estimators=[("mlpc", clf1), ('svc', clf2), ('qda', clf3)],
                                    voting='soft',
                                    weights=[mlp_weight, svc_weight, qda_weight])

            # Train and predict
            clf1.fit(X_train, y_train)
            clf2.fit(X_train, y_train)
            clf3.fit(X_train, y_train)
            eclf.fit(X_train, y_train)
            y_predict_eclf = eclf.predict(X_test)
            y_predict_mlp = clf1.predict(X_test)
            y_predict_svc = clf2.predict(X_test)
            y_predict_qda = clf3.predict(X_test)
            
            (rmse, mae, r2) = eval_metrics(y_test, y_predict_eclf)
            accuracy_eclf = accuracy_score(y_test, y_predict_eclf)
            accuracy_mlp = accuracy_score(y_test, y_predict_mlp)
            accuracy_svc = accuracy_score(y_test, y_predict_svc)
            accuracy_qda = accuracy_score(y_test, y_predict_qda)
            precision = precision_score(y_test, y_predict_eclf, average=None)
            f1 = f1_score(y_test, y_predict_eclf, average='weighted')
            print(f"Accuracy eclf: {accuracy_eclf}")
            print(f"Accuracy mlp: {accuracy_mlp}")
            print(f"Accuracy svc: {accuracy_svc}")
            print(f"Accuracy qda: {accuracy_qda}")
            print(f"Precision: {precision}")
            print(f"Root mean square error: {rmse}")
            print(f"R2 Coefficient determinator: {r2}")
            print(f"Mean absolute error: {mae}")
            print(f"F1 score: {f1}")
            cm_eclf = confusion_matrix(y_test, y_predict_eclf)
            cm_mlp = confusion_matrix(y_test, y_predict_mlp)
            cm_svc = confusion_matrix(y_test, y_predict_svc)
            cm_qda = confusion_matrix(y_test, y_predict_qda)
            # Display accuracy along with score and confusion matrix
            disp_eclf = ConfusionMatrixDisplay(confusion_matrix=cm_eclf, display_labels=letter_dict.keys())
            disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=letter_dict.keys())
            disp_svc = ConfusionMatrixDisplay(confusion_matrix=cm_svc, display_labels=letter_dict.keys())
            disp_qda = ConfusionMatrixDisplay(confusion_matrix=cm_qda, display_labels=letter_dict.keys())
            disp_eclf.plot()
            plt.show()
            disp_mlp.plot()
            plt.show()
            disp_svc.plot()
            plt.show()
            disp_qda.plot()
            plt.show()
            
            mlflow.log_param("SVC kernel", kernel)
            mlflow.log_param("SVC probability", probability)
            mlflow.log_param("SVC degree", degree)
            mlflow.log_param("SVC C", C)
            mlflow.log_param("MLP Solver", solver)
            mlflow.log_param("MLP activation", activation)
            mlflow.log_param("MLP learning rate", learning_rate)
            mlflow.log_param("MLP alpha", alpha)
            mlflow.log_param("MLP learning rate init", learning_rate_initialize)
            mlflow.log_param("MLP power t", power_t)
            mlflow.log_param("QDA tolerance", tolerance)
            mlflow.log_param("QDA Weight", qda_weight)
            mlflow.log_param("SVC Weight", svc_weight)
            mlflow.log_param("MLP Weight", mlp_weight)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("f1 score", f1)
            mlflow.log_metric("Voting classifier Accuracy", accuracy_eclf)
            mlflow.log_metric("MLP Accuracy", accuracy_mlp)
            mlflow.log_metric("SVC Accuracy", accuracy_svc)
            mlflow.log_metric("QDA Accuracy", accuracy_qda)
            
            signature = infer_signature(X_train, y_predict_eclf)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    eclf, "model", registered_model_name="Voting classifier SVC+MLP+QDA", signature=signature
                )
            else:
                mlflow.sklearn.log_model(eclf, "model", signature=signature)
        
        # Save data if flags are set and file does not exist
        if save_model and not os.path.isfile(filename):
            print("-- Saving model --")
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