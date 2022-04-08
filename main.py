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
    """loads dataset from scikitlearn toy datasets"""

    if dataset == "Iris Plants":
        data = datasets.load_iris()
    elif dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    x = data.data
    y = data.target

    return x, y

# TODO: Add more parameters
def add_model_parameter_ui(classifier):
    params = dict()
    if classifier == "KNN":
        k = st.sidebar.slider("K", 1, 15)
        weights = st.sidebar.radio("Weights", ["Uniform", "Distance"])
        algorithm = st.sidebar.radio('Algorithm', ['Auto', 'Ball-tree', 'KD-tree', 'Brute'])
        params["K"] = k
        params["Weights"] = weights.lower()
        params["Algorithm"] = algorithm.lower()
    elif classifier == "SVM":
        c = st.sidebar.slider("C", 0.01, 10.0)
        kernel = st.sidebar.radio('Kernel', ['Linear', 'Poly', 'RBF', 'Sigmoid'])
        params["C"] = c
        params["Kernel"] = kernel.lower()
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


def get_classifier(classifier, params):
    if classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"], weights=params["Weights"], algorithm=params["Algorithm"])
    elif classifier == "SVM":
        clf = SVC(C=params["C"], kernel=params["Kernel"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"], random_state=1)
    return clf


if __name__ == '__main__':
    # App title
    st.title("Machine Learning Classifiers")
    st.write('''
    Play with different datasets and machine learning classifiers to understand data better!
    ''')
    st.markdown('<hr>', unsafe_allow_html=True)

    # Sidebar
    dataset = st.sidebar.selectbox("Select Dataset", ["Iris Plants", "Breast Cancer", "Wine"])
    # TODO: Add these classifiers: Decision trees, Adaboost
    classifier = st.sidebar.selectbox("Select Classifier",
                                      ["KNN", "SVM", "Random Forest"])

    # Dataset Section
    data_section = st.container()
    data_section.header("Information about the dataset")
    X, y = get_dataset(dataset)
    # TODO: Display the list of features
    data_section.write("Number of Samples: ")
    data_section.write(len(X))
    data_section.write("Number of Features: ")
    data_section.write(np.shape(X)[1])
    data_section.write("Number of Classes: ")
    data_section.write(len(np.unique(y)))
    data_section.markdown('<hr>', unsafe_allow_html=True)

    # Model Section
    params = add_model_parameter_ui(classifier)
    clf = get_classifier(classifier, params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_section = st.container()
    model_section.header("Information about the classifier")
    model_section.write("Accuracy: ")
    model_section.write(acc)
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
    model_section.pyplot(fig)
