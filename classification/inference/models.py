from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.utils import to_categorical

# models need to be modified for specific use cases
def predict(X_train, y_train, X_test):
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train, y_train)

    return clf.predict(X_test)

def keras_model(embedding_matrix, embedding_size, input_length):
    # 1/27/19 Adapted from https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    # define model
    model = Sequential()
    e = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=input_length, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # summarize the model
    #print(model.summary())
    return model
