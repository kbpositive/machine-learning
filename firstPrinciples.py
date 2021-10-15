def tensorProduct(x, y):
    return [[a * b for b in y] for a in x]


def dotProduct(x, y):
    return sum([x[i] * y[i] for i, _ in enumerate(x)])


def vectorSubtraction(x, y):
    return [x[i] - y[i] for i, _ in enumerate(x)]


""" i = [1.0 for _ in range(5)]
w = [[0.1 * (j * i) for j in range(4)] for i in range(5)]
t = [0.0, 0.0, 1.0, 0.0]
a = 0.1

for _ in range(1000):
    o = [dotProduct(i, n) for n in [*map(list, zip(*w))]]
    u = tensorProduct([n * a for n in i], vectorSubtraction(o, t))
    w = [vectorSubtraction(w[n], u[n]) for n, _ in enumerate(u)] """

data = []
with open("./IrisClassifier/irisData.txt") as text:
    c = 0
    for line in text:
        data.append(
            [
                [float(i) for i in line.split(",")[:-1]],
                [1.0 if i == (c // 50) else 0.0 for i in range(3)],
            ]
        )
        c += 1


def normalize(x):
    t = [*zip(*x)]
    means = [min(col) for col in t]
    std = [max(col) - min(col) for index, col in enumerate(t)]
    return [[(n[m] - means[m]) / std[m] for m, _ in enumerate(n)] for n in x]


inputs = normalize([n[0] for n in data])
targets = [n[1] for n in data]
weights = [[0.001 * ((j + i) % 10) for j in range(1, 4)] for i in range(1, 151)]
learningRate = 0.01

# loop
for _ in range(2000):
    outputs = [[dotProduct(i, n) for n in [*map(list, zip(*weights))]] for i in inputs]
    u = [
        tensorProduct(
            [n * learningRate for n in inputs[index]],
            vectorSubtraction(o, targets[index]),
        )
        for index, o in enumerate(outputs)
    ]
    uSum = [
        [sum([u[c][a][b] for c, _ in enumerate(u)]) for b, _ in enumerate(u[0][0])]
        for a, _ in enumerate(u[0])
    ]
    weights = [vectorSubtraction(weights[n], uSum[n]) for n, _ in enumerate(uSum)]

for output in range(3):
    print([sum(n) / 50.0 for n in zip(*outputs[output * 50 : output * 50 + 50])])
