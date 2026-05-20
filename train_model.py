import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from lstm_model import create_model

X = np.load("C:/Users/saiva/Downloads/Rough/results/X_data.npy")
y = np.load("C:/Users/saiva/Downloads/Rough/results/y_data.npy")

# convert labels to one-hot
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = create_model()

model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=16,
    validation_data=(X_test, y_test)
)

model.save("C:/Users/saiva/Downloads/Rough/models/activity_model.keras")

print("Model trained successfully")