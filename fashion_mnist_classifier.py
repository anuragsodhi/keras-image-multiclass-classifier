from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# Splitting train to make val set and normalizing train set
length_val = int(X_train_full.shape[0] * 0.1)
X_valid, X_train = X_train_full[:length_val] /255.0 , X_train_full[length_val:] /255.0
y_valid, y_train = y_train_full[:length_val], y_train_full[length_val:]

# Adding class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Build sequential model
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=X_train_full.shape[1:]))
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))

#Summarizing
print(model.summary())

#Sample weights 1 dense layer (random)
print("weights", model.layers[1].get_weights()[0])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Training the model
hist = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# Plotting the fit
df = pd.DataFrame(hist.history)
df.plot()
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# Evaluate the model on test
y_pred = model.predict_classes(X_test)

# First 3 predictions
print(np.array(class_names)[y_pred[:3]])

##Finshed

