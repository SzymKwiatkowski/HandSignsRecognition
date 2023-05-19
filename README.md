# Hand signs recognition project
Project is dedicated to recognizing hand signs with simple machine learning application.

# Preparing data
While preparing first data batch for training `lookup_data.ipynb` notebook was created to fastly look into some details concerning simple data processing and vizualization. While using notebook some of the things were observed. First was data distribution of features in dataset:
<p align="center">
<img src="imgs/data_distribution.png" alt="Distribution of features" style="height: 300px; width:300px;"/>
<img src="imgs/test_data_distribution.png" alt="Distribution of features in test dataset" style="height: 300px; width:300px;"/>
<img src="imgs/train_data_distribution.png" alt="Distribution of features in test dataset" style="height: 300px; width:300px;"/>
</p>

# Model selection
Model that were tried during testing phase:
- VotingClassifier,
- MLPClassifier, 
- MLPRegressor, 
- DecisionTreeClassifier,
- LinearSVC,
- KNeighborsClassifier,
- MLPClassifier,
- MLPRegressor,
- SVR,
- SVC,
- DecisionTreeRegressor,
- BaggingClassifier,
- AdaBoostClassifier,
- QuadraticDiscriminantAnalysis
- ExtraTreesClassifier.

Voting classifier as well as other ensemble classifiers were tested, the same goes for boost classifiers.

## Currently best implemented solution:
Current best solution is consisting of voting classifiers consisting of QuadraticDiscriminantAnalysisClassifier and neural network tuned for application. This model currently has 95.6% accuracy score

# Test results
The best combination of classifiers was:
- Using VotingClassifier with QuadraticDiscriminantAnalysis and MLPClassifier
- The reason is that they compliment their Decision boundaries very well
- They also fit altogether into data way better than any other used combination even when trying 3 or 4 different ensembled classifiers

Some of the major things to mention in terms of every step of implementation is listed below. Every single one is using voting classifier

## First model created with 81% accuracy
<p align="center">
<img src="imgs/confusion_matrix_first_ensembled_model.png" alt="81% accuracy score confusion matrix" style="height: 500px; width:700px;"/>
</p>

As seen on graphic above matrix displays high error in terms of identifing `u` and `v` letter which is was very problematic and has dedicated segment of experimentation in notebook. Implementation consists of `Voting classifier` with `MLPClassifier` and `Decision tree`.

## Second model created with 91% Accuracy 
<p align="center">
<img src="imgs/confusion_matrix_first_ensembled_model_91per.png" alt="91% accuracy score confusion matrix" style="height: 500px; width:700px;"/>
</p>


As seen on graphic above matrix displays high error in terms of identifing `u` and `v` which is the same problem as above but not as extensive because error was sucessfully reduced. The problem still heavily persists which means that additional implementation of `Voting classifier` with `MLPClassifier` and `Polyminal SVC`. This means that problem cannot be solved via extending decision boundaries in *square like shapes* as in case of decision tree and some other classifier such as SVC, KNClassifier or QDA should be a huge help.

## Final model with 96% accuracy score
<p align="center">
<img src="imgs/voting_class_last.png" alt="96% accuracy score confusion matrix" style="height: 500px; width:700px;"/>
</p>

Error is successfully reduced and is not as big problem as previosly. Modification made is adding `QuadraticDiscriminantAnalysis` to voting classifier. This allowed to heavily reduce influence of previously wrongly detected `u` and `v` to very big extent.
