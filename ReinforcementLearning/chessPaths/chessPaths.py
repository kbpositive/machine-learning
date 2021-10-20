import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio
import os
import random


class Board:
    def __init__(self, rewards):
        self.rewards = rewards
        self.dims = np.array([len(self.rewards), len(self.rewards[0])])

    def state(self, pos):
        return np.eye(self.dims[0] * self.dims[1])[pos[0] * 8 + pos[1]]

    def reward(self, pos):
        return self.rewards[pos[0]][pos[1]]

    def get_states(self):
        return np.array(
            [
                self.state(np.array([row, col]))
                for row in range(self.dims[0])
                for col in range(self.dims[1])
            ]
        )

    def valid_move(self, x, h):
        return (
            x + h
            if (0 <= (x + h)[0] < self.dims[0] and 0 <= (x + h)[1] < self.dims[1])
            else x
        )


class Piece:
    def __init__(self, moves):
        self.discount = 0.95
        self.moves = self.makeMoves(moves)
        self.model = self.makeNet(len(self.moves))

    def makeMoves(self, initMoves):
        moves = [[0, 0]]
        for _ in range(4):
            for index, [row, col] in enumerate(initMoves):
                moves.append([row, col])
                initMoves[index][0], initMoves[index][1] = (
                    initMoves[index][1],
                    -initMoves[index][0],
                )
        return moves

    def makeNet(self, length):
        model = models.Sequential(
            [
                layers.Dense(length, input_shape=(64,), activation="sigmoid"),
            ]
        )
        optimizer = optimizers.Adam(learning_rate=0.02, amsgrad=True)
        model.compile(
            loss=self.reinforce,
            optimizer=optimizer,
            metrics=tf.keras.metrics.MeanAbsoluteError(),
        )
        return model

    def reinforce(self, actual, pred):
        return -((actual) * tf.math.log(pred))

    def policy(self, x):
        return self.model(np.array([board.state(x)]))[0]

    def next_state(self, board, moves, x):
        return np.array(board.valid_move(x, moves[random.choices(range(len(moves)), k=1, weights=self.policy(x))[0]]))

    def rollout(self, board, moves, state, timesteps):
        p = random.choices(range(len(moves)), k=1, weights=self.policy(state))[0]

        states = [np.array(board.valid_move(state, moves[p]))]
        for i in range(1,timesteps):
            states.append(self.next_state(board, moves, states[-1]))

        
        advantage = 0.0
        for i in range(len(states)-1,-1,-1):
            advantage = (advantage + board.reward(states[i]) - self.policy(states[i])) * (self.discount ** (i))
        
        return (advantage*np.eye(len(moves))[p]) + np.array(
                [board.reward(board.valid_move(state, action)) for action in moves]
            ) - (self.policy(state))


class King(Piece):
    def __init__(self):
        super().__init__([[0, -1], [-1, -1]])
        self.label = "King"


class Knight(Piece):
    def __init__(self):
        super().__init__([[-1, -2], [-2, -1]])
        self.label = "Knight"


class Bishop(Piece):
    def __init__(self):
        super().__init__([[-n, -n] for n in range(1, 8)])
        self.label = "Bishop"


class Rook(Piece):
    def __init__(self):
        super().__init__([[-n, 0] for n in range(1, 8)])
        self.label = "Rook"


class Queen(Piece):
    def __init__(self):
        super().__init__(
            [[-n, -n] for n in range(1, 8)] + [[-n, 0] for n in range(1, 8)]
        )
        self.label = "Queen"


def render(board, piece, result, epoch):
    # plt 1
    sns.set(rc={"figure.figsize": (9, 3)})
    sns.set_style("whitegrid")
    _, axs = plt.subplots(ncols=3)
    sns.lineplot(
        data=np.array(result),
        ax=axs[0],
    )
    axs[0].legend_.remove()
    axs[0].set_title("Mean Absolute Error", fontsize=8)
    plt.setp(axs[0].lines, color="#699CB3", linewidth=0.75)

    # plt 2
    sns.lineplot(
        data=piece.model.predict(board.get_states()),
        palette=sns.color_palette(f"dark:#347893_r", len(piece.moves)),
        dashes={n: "" for n in range(len(piece.moves))},
        ax=axs[1],
    )
    axs[1].legend_.remove()
    axs[1].set_title("Policy value by board state", fontsize=8)
    plt.setp(axs[1].lines, linewidth=0.75)

    # plt 3
    heatmap_data = np.array(
        [np.mean(i) for i in piece.model.predict(board.get_states())]
    ).reshape((8, 8))
    sns.heatmap(
        data=heatmap_data,
        ax=axs[2],
        cbar=False,
        cmap=sns.light_palette("#205565", as_cmap=True, reverse=True),
    ).invert_yaxis()
    axs[2].set_title("Mean value by board state", fontsize=8)

    plt.savefig(f"./results/{epoch}.png")
    plt.close()
    return f"./results/{epoch}.png"


def saveGif(label, files):
    with imageio.get_writer(f"./{label}.gif", mode="I") as writer:
        for file in files:
            image = imageio.imread(file)
            writer.append_data(image)
        for _ in range(36):
            image = imageio.imread(files[-1])
            writer.append_data(image)

    for file in set(files):
        os.remove(file)


def training_loop(board, piece, loops, timesteps):
    result = []
    files = []
    for epoch in range(loops):
        out = (
            np.array(
                [
                    np.mean([piece.rollout(
                        board,
                        [piece.moves[n] for n in range(len(piece.moves))],
                        np.array([row, col]),
                        timesteps,
                    ) for _ in range(1)], axis=0)
                    for row in range(board.dims[0])
                    for col in range(board.dims[1])
                ]
            )
        )

        history = piece.model.fit(
            board.get_states(),
            out,
            epochs=1,
            verbose=0,
        )

        result.append(history.history[[i for i in history.history.keys()][-1]])

        files.append(render(board, piece, result, epoch))

    saveGif(piece.label, files)


def make_board(rows, cols):
    board = Board(np.zeros((rows, cols))+0.5)
    board.rewards[1][1] = 1.0
    board.rewards[2][2] = 0.0
    board.rewards[1][2] = 0.0
    board.rewards[2][1] = 0.0
    board.rewards[6][6] = 1.0
    board.rewards[5][5] = 0.0
    board.rewards[6][5] = 0.0
    board.rewards[5][6] = 0.0
    return board


if __name__ == "__main__":
    board = make_board(8, 8)
    training_loop(board, King(), 300, 8)
