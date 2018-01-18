if True:

    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X,y = datasets.load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = DecisionTreeClassifier(min_samples_split=8, min_samples_leaf=4)

    model.fit(X_train, y_train)

    y_predicted = model.predict(X_test)

    print("Score: %.4f " % accuracy_score(y_test, y_predicted))
