import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import imageio
import os

from tensorflow.python.keras.callbacks import History


class bandit:
    def __init__(self, arms):
        self.arms = arms
        self.actions = np.random.uniform(0.0, 1.0, arms)
        self.state = np.ones(arms)
        self.model = models.Sequential(
            [layers.Dense(arms, input_shape=(arms,), activation="softmax")]
        )
        self.optimizer = optimizers.Adam(learning_rate=0.005)
        self.model.compile(
            loss=self.reinforce,
            optimizer=self.optimizer,
            metrics=tf.keras.metrics.MeanAbsoluteError(),
        )

    def pull(self):
        return np.array(
            [
                (0.0) ** int(action < np.random.uniform(0.0, 1.0))
                for action in self.actions
            ]
        )

    def reinforce(self, actual, pred):
        return pred - actual * tf.math.log(pred)


if __name__ == "__main__":
    arms = 4
    con = bandit(arms)
    result = []
    files = []

    for epoch in range(100):
        history = con.model.fit(
            np.array([con.state]),
            tf.nn.softmax(
                np.array([np.mean([con.pull() for _ in range(500)], axis=0)])
            ),
            epochs=1,
            verbose=0,
        )

        result.extend(history.history[[i for i in history.history.keys()][-1]])
        fig, axs = plt.subplots(ncols=2)
        sns.lineplot(data=result, color="#8FCACA", ax=axs[0])
        sns.lineplot(
            data=[
                con.model.predict(np.array([con.state]))[0],
                tf.nn.softmax(con.actions),
            ],
            palette={0: "#8FCACA", 1: "#FFAEA5"},
            dashes={0: "", 1: ""},
            ax=axs[1],
        )
        files.append(f"./nArmedBandits/results/{epoch}.png")
        plt.savefig(files[-1])
        plt.close()

    with imageio.get_writer("./nArmedBandits/results.gif", mode="I") as writer:
        for file in files:
            image = imageio.imread(file)
            writer.append_data(image)
        for _ in range(36):
            image = imageio.imread(files[-1])
            writer.append_data(image)

    for file in set(files):
        os.remove(file)
