# Created by Sijmen van der Willik
# 24/07/2020 12:14

import numpy as np


class Answer:
    def __init__(self):
        self.answers = []

    def add(self, x):
        self.answers.append(x)

    def reveal(self):
        return "Answers: " + "\n".join(self.answers)


def gen_target(data: np.array, n_lin=1):
    """Generate target column that can be derived from data"""
    # linear correlations
    y = np.zeros(data.shape[0]) + np.random.randint(-100, 1000)

    predictors = np.random.choice(data.shape[1], n_lin)

    answer = Answer()

    for p in predictors:
        corr_strength = 0.3 + np.random.random() * 0.7
        y += data[:, p] * corr_strength
        answer.add(f"Col {p} adds linearly to target with factor {corr_strength}")

    # return as column
    return y.reshape((-1, 1)), answer


def uniform_dist(size: int):
    low = np.random.randint(-100, 1000)
    high = low + np.random.randint(0, 1000)
    return np.random.uniform(low, high, size)


def norm_dist(size: int):
    mu = np.random.randint(-100, 200)
    sigma = np.random.random() ** 2 * np.random.randint(0, 100)
    return np.random.normal(mu, sigma, size)


def add_insights(data: np.array):
    cols = np.arange(data.shape[1])
    np.random.shuffle(cols)

    # put in given distributions
    # select columns
    norm_fraction = 0.5
    n_norm = int(norm_fraction * data.shape[1])

    # add distributions
    for c in cols[:n_norm]:
        data[:, c] = norm_dist(data.shape[0])
    for c in cols[n_norm:]:
        data[:, c] = uniform_dist(data.shape[0])

    # generate target column
    target, answer = gen_target(data)

    # destroy values in X
    missing_vals = True
    missing_fraction = 0.01
    if missing_vals:
        mask = np.random.random(data.shape) < missing_fraction
        data[mask] = np.nan

    # add target column
    data = np.hstack((data, target))

    return data, answer


def generate_data(size=100, n_vars=4):
    data = np.zeros((size, n_vars))

    # insights
    data, answer = add_insights(data)

    # return as X, y
    X = data[:, :-1]
    y = data[:, -1]

    return X, y, answer


if __name__ == "__main__":
    X, y, answer = generate_data()
    np.save("X", X)
    np.save("y", y)
    with open("./answers.txt", 'w') as f:
        f.write(answer.reveal())
