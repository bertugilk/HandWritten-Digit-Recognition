import keras
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, BatchNormalization, Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.datasets import mnist
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_of_trainImgs = x_train.shape[0]  # 60000 here
num_of_testImgs = x_test.shape[0]  # 10000 here

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(num_of_trainImgs, 1, img_rows, img_cols)
    x_test = x_test.reshape(num_of_testImgs, 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(num_of_trainImgs, img_rows, img_cols, 1)
    x_test = x_test.reshape(num_of_testImgs, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Convolution2D(32,3, 3 ,activation='relu',input_shape=input_shape))
model.add(BatchNormalization())

model.add(Convolution2D(64, 3, 3 ,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

model.fit(x_train,y_train,
          batch_size=128,
          epochs=15,
          verbose=1,
          validation_data=(x_test,y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0]) # %95 accuracy
print('Test accuracy:', score[1])

model.save("Model/model.h5")