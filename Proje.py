import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

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
    print("Corralation")
    correlation = data.corr()
    print(correlation)
    print("Loading Heatmap...")
    plt.figure(figsize=(15, 15))
    sns.heatmap(correlation, annot=True, fmt=".0%")
    # plt.show()

    for column in data.columns:
        if data[column].dtype == np.number:
            continue
        data[column] = LabelEncoder().fit_transform(data[column])
    data["Age_Years"] = data["Age"]
    data = data.drop("Age", axis=1)

    X = data.drop('Outcome', axis=1)
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

    for i in range(1, 20):
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
    
     kmeans = KMeans(n_clusters=3)
    k_fit = kmeans.fit(X)
    kumeler = k_fit.labels_
    plt.scatter(X[:, 0], X[:, 1], c=kumeler, s=40, cmap='viridis');
    identified_clusters = kmeans.fit_predict(X)

    labels = kmeans.fit(X).predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

    mlpc = MLPClassifier().fit(X_train,y_train)
    y_pred = mlpc.predict(X_test)
    accuracy_score(y_test,y_pred)


    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)



def main():
    csv_file = pd.read_csv("diabetes.csv")
    get_dataset_informatin(csv_file=csv_file)


if __name__ == '__main__':
    main()
