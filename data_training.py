import os
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not (i.split(".")[0] == "labels"):
        if not (is_init):
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c = c + 1

for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]

# Ensure y is converted to integer labels
y = y.astype(int)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.flatten())  # Flatten y to make it 1D
y = to_categorical(y_encoded)

X_new = X.copy()
y_new = y.copy()
counter = 0

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter = counter + 1

# Define CNN input layer
p = Input(shape=(X.shape[1],))

m = Dense(512, activation="relu")(p)
m = Dense(256, activation="sigmoid")(m)

op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=p, outputs=op)
#model compilation 
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])

tb_callback1 = tf.keras.callbacks.TensorBoard(log_dir="./tf_logs", histogram_freq=1)

model.fit(X, y, epochs=60, callbacks=[tb_callback1])
loss, accuracy = model.evaluate(X, y)

print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

#showing confusion matrics

y_pred = model.predict(X_new)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_new, axis=1)
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
sn.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

model.save("model.h5")
np.save("labels.npy", np.array(label))
