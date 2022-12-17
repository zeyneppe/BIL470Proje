import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lime import lime_tabular
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def get_dataset_informatin(csv_file="diabetes.csv"):
    data = csv_file
    print(data.head())
    print("Shaper Of Data: ", data.shape)
    print("Feature ------------------------ Type")
    print(data.dtypes)
    for column in data.columns:
        if data[column].dtype == object:
            print(str(column) + " : " + str(data[column].unique()))
            print(data[column].value_counts())
            print("________________________________________________________")
    print("Correlation")
    correlation = data.corr()
    print(correlation)
    print("Loading Heatmap...")
    plt.figure(figsize=(15, 15))
    sns.heatmap(correlation, annot=True, fmt=".0%")
    plt.show()

    for column in data.columns:
        if data[column].dtype == np.number:
            continue
        data[column] = LabelEncoder().fit_transform(data[column])
    data["Age_Years"] = data["Age"]
    data = data.drop("Age", axis=1)

    X = data.drop('Outcome', axis=1).drop('Pregnancies', axis=1)
    y = data.Outcome
    # X = data.iloc[:, 1:data.shape[1]].values
    # y = data.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345, shuffle=True)

    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
    scores = cross_val_score(pipeline, X=X_train, y=y_train, cv=10, n_jobs=5)
    print('Cross Validation accuracy scores on Train: %s' % scores)
    print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    forest = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
    forest.fit(X_train, y_train)
    print("Accuracy:", forest.score(X_train, y_train) * 100)
    cm = confusion_matrix(y_test, forest.predict(X_test))

    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]

    print(cm)
    print((TP + TN) / (TP + TN + FP + FN))

    model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))
    print(classification_report(y_test, y_pred))

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    print("*************************")
    print(svclassifier.score(X_train, y_train))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    knnTrainScores = []
    knnTestScores = []

    for i in range(1, 30):
        knnClassifier = KNeighborsClassifier(i)
        knnClassifier.fit(X_train, y_train)

        knnTrainScores.append(knnClassifier.score(X_train, y_train))
        knnTestScores.append(knnClassifier.score(X_test, y_test))

    bestTrainIndex = knnTrainScores.index(max(knnTrainScores))
    bestTestIndex = knnTestScores.index(max(knnTestScores))

    print(
        'Max train score {} with k = {} and test score = {}'.format(knnTrainScores[bestTrainIndex], bestTrainIndex + 1,
                                                                    knnTestScores[bestTrainIndex]))
    print('Max test score {} with k = {} and train score = {}'.format(knnTestScores[bestTestIndex], bestTestIndex + 1,
                                                                      knnTrainScores[bestTestIndex]))
    knnClassifier = KNeighborsClassifier(bestTestIndex + 1)
    knnClassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print(classification_report(y_test, y_pred))

    gradientBoostingClassifier = GradientBoostingClassifier()
    gradientBoostingClassifier.fit(X_train, y_train)
    y_pred = gradientBoostingClassifier.predict(X_test)
    print("\nTrain Accuracy with GradientBoostingClassifier: ", gradientBoostingClassifier.score(X_train, y_train))
    print("Test Accuracy with GradientBoostingClassifier:  ", gradientBoostingClassifier.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    histogramGradientClassifier = HistGradientBoostingClassifier()
    histogramGradientClassifier.fit(X_train, y_train)
    y_pred = histogramGradientClassifier.predict(X_test)
    print("\nTrain Accuracy with HistGradientBoostingClassifier: ", histogramGradientClassifier.score(X_train, y_train))
    print("Test Accuracy with HistGradientBoostingClassifier:  ", histogramGradientClassifier.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    adaBoostClassifier = AdaBoostClassifier()
    adaBoostClassifier.fit(X_train, y_train)
    y_pred = adaBoostClassifier.predict(X_test)
    print("\nTrain Accuracy with AdaBoostClassifier: ", adaBoostClassifier.score(X_train, y_train))
    print("Test Accuracy with AdaBoostClassifier:  ", adaBoostClassifier.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    randomForestClassifier = RandomForestClassifier(random_state=12345)
    randomForestClassifier.fit(X_train, y_train)
    y_pred = randomForestClassifier.predict(X_test)
    print("\nTrain Accuracy with RandomForestClassifier: ", randomForestClassifier.score(X_train, y_train))
    print("Test Accuracy with RandomForestClassifier:  ", randomForestClassifier.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    decisionTreeClassifier = DecisionTreeClassifier(random_state=12345)
    decisionTreeClassifier.fit(X_train, y_train)
    y_pred = decisionTreeClassifier.predict(X_test)
    print("\nTrain Accuracy with DecisionTreeClassifier: ", decisionTreeClassifier.score(X_train, y_train))
    print("Test Accuracy with DecisionTreeClassifier:  ", decisionTreeClassifier.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_train, y_train)
    y_pred = kmeans.predict(X_test)
    print("\nAccuracy with KMeans: ", accuracy_score(y_test, kmeans.predict(X_test)))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    mlpClassifier = MLPClassifier()
    mlpClassifier.fit(X_train, y_train)
    y_pred = mlpClassifier.predict(X_test)
    print("\nTrain Accuracy with MLPClassifier: ", mlpClassifier.score(X_train, y_train))
    print("Test Accuracy with MLPClassifier:  ", mlpClassifier.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    pca = PCA(n_components=2)
    X_train2 = pca.fit_transform(X_train)
    X_test2 = pca.transform(X_test)
    logisticRegression = LogisticRegression(random_state=12345)
    logisticRegression.fit(X_train2, y_train)
    y_pred = logisticRegression.predict(X_test2)
    print("\nTrain Accuracy with LogisticRegression with PCA: ", logisticRegression.score(X_train2, y_train))
    print("Test Accuracy with LogisticRegression with PCA:  ", logisticRegression.score(X_test2, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    explainer = shap.TreeExplainer(gradientBoostingClassifier)
    shap_values = explainer.shap_values(X_train)

    explainer = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values.tolist(),
                                                  class_names=[0, 1], verbose=True, mode='classification')

    print("Predicted Value: " + str(adaBoostClassifier.predict(X_test.values)[0]))
    exp = explainer.explain_instance(
        data_row=X_test.values[0],
        predict_fn=adaBoostClassifier.predict_proba,
        num_features=7
    )

    exp.show_in_notebook(show_table=True)

    print("Predicted Value: " + str(adaBoostClassifier.predict(X_test.values)[217]))
    exp = explainer.explain_instance(
        data_row=X_test.values[217],
        predict_fn=adaBoostClassifier.predict_proba,
        num_features=7
    )

    exp.show_in_notebook(show_table=True)

    explainer = shap.Explainer(gradientBoostingClassifier, X_train)
    shap_values = explainer(X_train)

    # contribution of each feature moves the value from the expected model output over
    # the background dataset to the model output for this prediction
    shap.plots.waterfall(shap_values[25])
    plt.show()


def main():
    csv_file = pd.read_csv("diabetes.csv")
    get_dataset_informatin(csv_file=csv_file)


if __name__ == '__main__':
    main()
