import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v1 import SGD
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

X_train = pd.read_csv(r"data/spiral_data_train.csv").values
y_train = pd.read_csv(r"data/spiral_labels_train.csv").values.ravel()
X_test = pd.read_csv(r"data/spiral_data_test.csv").values

def expand_features(X):
    X1, X2 = X[:, 0], X[:, 1]
    return np.column_stack([
        X1,
        X2,
        X1**2,
        X2**2,
        # X1*X2,
        np.sin(X1),
        np.sin(X2)
    ])

X_train_full = expand_features(X_train)
X_test_full = expand_features(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test_full)

model = Sequential()
model.add(Dense(6, activation='sigmoid', input_shape=(X_train.shape[1], )))
model.add(Dense(1, activation='sigmoid'))

optimizer = SGD(lr=0.1)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=20, min_delta=1e-4, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, batch_size=10, verbose=1, callbacks=[early_stop])

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

print(f"Training loss: {train_loss:.3f}, Validation loss: {val_loss:.3f}")
print(f"Training accuracy: {train_acc:.3f}, Validation accuracy: {val_acc:.3f}")

y_pred = (model.predict(X_test) > 0.5).astype(int)
np.savetxt("spiral_predictions.csv", y_pred, fmt="%d", delimiter=",")
print("Predictions saved to spiral_predictions.csv")

import matplotlib.pyplot as plt
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', s=5)
plt.show()