from numpy import loadtxt
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
dataset = loadtxt(
    '../../training_sources/pima-indians-diabetes.data.csv', delimiter=',')
print(dataset)

X = dataset[:, 0:8]
y = dataset[:, 8]

# Split dataset to train, validation, test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42)

# Train
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.summary()

opt_sgd = tf.keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(
    optimizer=opt_sgd,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, batch_size=8, epochs=20,
          validation_data=(X_val, y_val), verbose=1, validation_split=0.2)
model.save('model.h5')

# model = load_model('model.h5')

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Loss: %.2f" % loss)
print("Accuracy: %.2f" % (acc * 100.0))

X_new = X_test[20]
y_new = y_test[20]

X_new = np.expand_dims(X_new, axis=0)

y_pred = model.predict(X_new)
print("Gia tri du doan la: ", y_pred)
print("Gia tri dung la: ", y_new)
