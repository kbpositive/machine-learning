import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# build model
model = Sequential()
model.add(Dense(4, activation="sigmoid", input_shape=(4,)))
model.add(Dense(3))
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.2),
    loss="mse",
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

# import and format data
irisData = []
irisLabels = []

with open("./irisData.txt", "r", encoding="utf-8") as data:
    for line in data:
        values = line.split(",")
        irisData.append([float(num) for num in values[:-1]])
        irisLabels.append(values[-1])

inputs = np.array(irisData)
nameCodes = [
    [1 if logit == label else 0 for logit in set(irisLabels)] for label in irisLabels
]
outputs = np.array(nameCodes)

# shuffle data
shuffledInputs = np.array(
    [inputs[(example * 50) % 150 + (example // 3)] for example, _ in enumerate(inputs)]
)
shuffledOutputs = np.array(
    [
        outputs[(example * 50) % 150 + (example // 3)]
        for example, _ in enumerate(outputs)
    ]
)

# inputs, outputs
x, y = shuffledInputs[:110], shuffledOutputs[:110]

# validation inputs, validation outputs
a, b = shuffledInputs[110:], shuffledOutputs[110:]

model.fit(
    x,
    y,
    batch_size=64,
    epochs=400,
    validation_data=(a, b),
)
# loss: ~0.0533 - categorical_accuracy: ~0.9455 - val_loss: ~0.0578 - val_categorical_accuracy: ~0.9500
