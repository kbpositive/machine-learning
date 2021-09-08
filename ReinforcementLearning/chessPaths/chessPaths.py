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
                layers.Dense(length, input_shape=(64,), activation="tanh"),
            ]
        )
        optimizer = optimizers.Adam(learning_rate=0.0375)
        model.compile(
            loss=self.reinforce,
            optimizer=optimizer,
            metrics=tf.keras.metrics.MeanAbsoluteError(),
        )
        return model

    def reinforce(self, actual, pred):
        pShift = lambda x: (x + 1.0) / 2.0
        return pShift(pred) - pShift(actual) * tf.math.log(pShift(pred))

    def rollout(self, board, moves, state, timesteps, depth=1):
        if (-1) >= timesteps:
            return 0

        policy = lambda x: self.model(np.array([board.state(x)]))[0]
        action = policy(state)
        state_shift = np.array(action + np.abs(np.min(action))) ** (
            timesteps + depth - 1
        )

        next_action = np.argmax(policy(state))
        guess = random.choices(moves, k=1, weights=state_shift)[0]
        if board.reward(state) == 1.0 or board.reward(state) == -1.0:
            next_action = 0
            guess = moves[0]

        acc_rewards = (
            self.rollout(
                board,
                moves,
                np.array(board.valid_move(state, guess)),
                timesteps - 1,
                depth + 1,
            )
            * (self.discount ** depth)
        )

        return (
            np.array(
                [board.reward(board.valid_move(state, action)) for action in moves]
            )
            + (np.eye(len(moves))[next_action] * acc_rewards)
            if depth == 1
            else board.reward(board.valid_move(state, guess)) + acc_rewards
        )


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
                    piece.rollout(
                        board,
                        [piece.moves[n] for n in range(len(piece.moves))],
                        np.array([row, col]),
                        timesteps,
                    )
                    for row in range(board.dims[0])
                    for col in range(board.dims[1])
                ]
            )
            / max(1, timesteps)
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
    board = Board(np.zeros((rows, cols)))
    board.rewards[1][1] = 1.0
    board.rewards[2][2] = -1.0
    board.rewards[1][2] = -1.0
    board.rewards[2][1] = -1.0
    board.rewards[6][6] = 1.0
    board.rewards[5][5] = -1.0
    board.rewards[6][5] = -1.0
    board.rewards[5][6] = -1.0
    return board


if __name__ == "__main__":
    board = make_board(8, 8)
    training_loop(board, King(), 150, 8)
