import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.utils import shuffle

path = r'C:\Users\arhyr\Desktop\self_driving_car\\'

with open(path+'test.p', mode='rb') as f:
    test = pickle.load(f)
with open(path+'train.p', mode='rb') as f:
    train = pickle.load(f)
with open(path+'valid.p', mode='rb') as f:
    valid = pickle.load(f)

X_train, y_train = train['features']/255.0, to_categorical(train['labels'])
X_valid, y_valid = valid['features']/255.0, to_categorical(valid['labels'])
X_test, y_test = test['features']/255.0, to_categorical(test['labels'])

X_train, y_train = shuffle(X_train, y_train)

n_train = y_train.shape[0]

n_validation = y_valid.shape[0]

n_test = y_test.shape[0]

image_shape = (X_train.shape[1], X_train.shape[2])

# n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
# print("Number of classes =", n_classes)

# num = randint(0, X_train.shape[0]-1)
# pic = X_train[num]
# plt.imshow(pic)
# plt.show()

model = Sequential([
    Conv2D(128, (5, 5), input_shape=(32, 32, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(43, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=256, validation_data=(X_valid, y_valid), epochs=50, verbose=2)


plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

# Plot history for loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')



