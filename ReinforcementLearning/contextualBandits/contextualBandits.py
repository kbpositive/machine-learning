import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import imageio
import os


class bandit:
    def __init__(self, context, arms):
        self.arms = arms
        self.context = context
        self.actions = np.array(
            [np.random.uniform(0.0, 1.0, arms) for _ in range(context)]
        )
        self.state = np.eye(context)
        self.model = models.Sequential(
            [layers.Dense(arms, input_shape=(arms,), activation="softmax")]
        )
        self.optimizer = optimizers.Adam(learning_rate=0.008)
        self.model.compile(
            loss=self.reinforce,
            optimizer=self.optimizer,
            metrics=tf.keras.metrics.MeanAbsoluteError(),
        )

    def pull(self, state):
        return np.array(
            [
                (0.0) ** int(action < np.random.uniform(0.0, 1.0))
                for action in self.actions[state]
            ]
        )

    def reinforce(self, actual, pred):
        return pred - actual * tf.math.log(pred)


if __name__ == "__main__":
    context = 4
    arms = 4
    con = bandit(context, arms)
    result = []
    files = []

    for epoch in range(200):
        history = con.model.fit(
            con.state,
            np.array(
                [
                    tf.nn.softmax(i)
                    for i in np.mean(
                        [
                            [con.pull(i) for i in range(len(con.actions))]
                            for _ in range(500)
                        ],
                        axis=0,
                    )
                ]
            ),
            epochs=1,
            verbose=0,
        )

        result.append(history.history[[i for i in history.history.keys()][-1]])
        fig, axs = plt.subplots(ncols=2)
        sns.lineplot(
            data=np.array(result),
            palette={0: "#8FCACA", 1: "#9FCACA", 2: "#AFCACA", 3: "#BFCACA"},
            dashes={0: "", 1: "", 2: "", 3: ""},
            ax=axs[0],
        )
        sns.lineplot(
            data=np.transpose(
                np.append(
                    con.model.predict(con.state),
                    [tf.nn.softmax(i) for i in con.actions],
                    axis=0,
                )
            ),
            palette={
                0: "#8FCACA",
                1: "#9FCACA",
                2: "#AFCACA",
                3: "#BFCACA",
                4: "#FFBEB5",
                5: "#FFCEC5",
                6: "#FFDED5",
                7: "#FFEEE5",
            },
            dashes={0: "", 1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: ""},
            ax=axs[1],
        )

        files.append(f"./contextualBandits/results/{epoch}.png")
        plt.savefig(files[-1])
        plt.close()

    with imageio.get_writer("./contextualBandits/results.gif", mode="I") as writer:
        for file in files:
            image = imageio.imread(file)
            writer.append_data(image)
        for _ in range(36):
            image = imageio.imread(files[-1])
            writer.append_data(image)

    for file in set(files):
        os.remove(file)