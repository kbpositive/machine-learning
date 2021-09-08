import operator
import random
import math

genome = {
    "0": [0.0, [0.0, 0.0, 0.0, 0.0, 0.0]],
    "1": [1.0, [0.0, 0.0, 0.0, 0.0, 1.0]],
    "2": [2.0, [0.0, 0.0, 0.0, 1.0, 0.0]],
    "3": [3.0, [0.0, 0.0, 0.0, 1.0, 1.0]],
    "4": [4.0, [0.0, 0.0, 1.0, 0.0, 0.0]],
    "5": [5.0, [0.0, 0.0, 1.0, 0.0, 1.0]],
    "6": [6.0, [0.0, 0.0, 1.0, 1.0, 0.0]],
    "7": [7.0, [0.0, 0.0, 1.0, 1.0, 1.0]],
    "8": [8.0, [0.0, 1.0, 0.0, 0.0, 0.0]],
    "9": [9.0, [0.0, 1.0, 0.0, 0.0, 1.0]],
    "+": [operator.add, [0.0, 1.0, 0.0, 1.0, 0.0]],
    "-": [operator.sub, [0.0, 1.0, 0.0, 1.0, 1.0]],
    "*": [operator.mul, [0.0, 1.0, 1.0, 0.0, 0.0]],
    "/": [operator.truediv, [0.0, 1.0, 1.0, 0.0, 1.0]],
}


def orgEval(organism):
    if (
        organism == ""
        or not type(genome[organism[0]][0]) == type(genome[organism[-1]][0]) == float
    ):
        return None
    opers = [0.0, []]
    sign = None
    for index, char in enumerate(organism):
        if type(genome[char][0]) != float:
            if not opers[1]:
                if sign:
                    return None
                continue
            elif not sign:
                sign = genome[char][0]
                opers[0] = float("".join(opers[1]))
                opers[1] = []
                continue
            elif type(genome[organism[index - 1]][0]) != float:
                return None
            opers[0] = (
                sign(opers[0], float("".join(opers[1])))
                if float("".join(opers[1])) != 0.0
                else 0.0
            )
            opers[1] = []
            sign = genome[char][0]
        else:
            opers[1].append(char)
    if opers[1] and sign:
        opers[0] = (
            sign(opers[0], float("".join(opers[1])))
            if float("".join(opers[1])) != 0.0
            else 0.0
        )
        return opers[0]
    return None


def fitness(organism, target):
    fit = 10.0
    if orgEval(organism) is None:
        fit -= 10.0
    else:
        fit -= math.log(1 + abs(target - orgEval(organism))) / 3
    return fit


def orgCreate(chromosome):
    organism = []
    for c in range(len(chromosome) // 5):
        for key, (value, gene) in genome.items():
            if gene == chromosome[c * 5 : (c + 1) * 5]:
                organism.append(key)
                continue
    return "".join(organism)


def orgSeed(numGenes):
    chromosome = random.choices([0.0, 1.0], k=5 * numGenes)
    return orgCreate(chromosome), chromosome


def mutate(chromosome):
    for _ in range(mutationRate):
        mutation = random.choice(range(len(chromosome)))
        chromosome[mutation] = random.choice([0.0, 1.0])
    return chromosome


def generateOffspring(chromosome, num):
    return [
        [orgCreate(mutate(chromosome[:])), mutate(chromosome[:])] for _ in range(num)
    ]


def prune(offspring, num, target):
    return sorted(
        [[fitness(child, target), child, c] for child, c in offspring], reverse=True
    )[:num]


target = 7919
chromosomeLength = 8
growthRate = 100
acceptanceRate = 5
mutationRate = 4

organism, chromosome = orgSeed(chromosomeLength)
offspring = generateOffspring(chromosome, growthRate)

mostFit = prune(offspring, acceptanceRate, target)
for _ in range(1000):
    total = []
    for group in mostFit:
        total.extend(generateOffspring(group[2], growthRate))

    mostFit = prune(total, acceptanceRate, target)

    print(mostFit[0][:2])
    if mostFit[0][0] >= 10:
        break
