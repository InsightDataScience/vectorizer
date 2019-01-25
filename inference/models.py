from sklearn.linear_model import LogisticRegression

# models need to be modified for specific use cases
def predict(X_train, y_train, X_test):
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train, y_train)

    return clf.predict(X_test)
