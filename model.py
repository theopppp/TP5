import tensorflow
import pandas as pd
import pickle
from tensorflow import keras

# import, prepare train/test, scale

mnist_train = pd.read_csv(r"C:\Users\theog\Downloads\data\fashion-mnist-train-1.csv")
mnist_test = pd.read_csv(r"C:\Users\theog\Downloads\data\fashion-mnist_test.csv")

y_train = mnist_train['label']
X_train = mnist_train.drop(columns=['label'])

y_test = mnist_test['label']
X_test = mnist_test.drop(columns=['label'])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

X_train = X_train / 255.0
X_test = X_test / 255.0

# create and train NN
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train.shape[1:])),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=35)

# save model with pickle
pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl', 'rb'))