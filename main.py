import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def get_dataset(dataset):
    if dataset == "Iris":
        data = datasets.load_iris()
    elif dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    x = data.data
    y = data.target
    return x, y


def add_model_parameter_ui(classifier):
    params = dict()
    if classifier == "KNN":
        k = st.sidebar.slider("K", 1, 15)
        params["K"] = k
    elif classifier == "SVM":
        c = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = c
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


def get_classifier(classifier, params):
    if classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"], random_state=1)
    return clf


if __name__ == '__main__':
    # App title
    st.title("Machine Learning Classifiers")
    st.write('''
    Play with different datasets and machine learning classifiers to understand data better!
    ''')
    # Sidebar
    dataset = st.sidebar.selectbox("Select Dataset", ["Iris", "Breast Cancer", "Wine"])
    classifier = st.sidebar.selectbox("Select Classifier", ["KNN", "SVM", "Random Forest"])
    # Section description
    data_section = st.container()
    data_section.header("Information about your dataset")
    # Dataset description
    X, y = get_dataset(dataset)
    data_section.write("Shape of Dataset: ")
    data_section.write(X.shape)
    data_section.write("Number of Classes: ")
    data_section.write(len(np.unique(y)))
    # Model description
    params = add_model_parameter_ui(classifier)
    clf = get_classifier(classifier, params)
    # Classification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    data_section.write("Classifier: ")
    data_section.write(classifier)
    data_section.write("Accuracy: ")
    data_section.write(acc)
    # Plotting
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    data_section.pyplot(fig)
