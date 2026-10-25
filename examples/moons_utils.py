import random

from tianyi_micrograd.nn import MLP


def make_data():
    from sklearn.datasets import make_moons

    xs, ys = make_moons(n_samples=100, noise=0.1, random_state=1337)
    return xs.tolist(), (2 * ys - 1).tolist()


def make_model():
    random.seed(1337)
    return MLP(2, [16, 16, 1])


def loss_and_accuracy(model, xs, ys, alpha=1e-4):
    scores = [model(x) for x in xs]
    losses = [(1 - y * score).relu() for y, score in zip(ys, scores)]
    data_loss = sum(losses) / len(losses)
    reg_loss = alpha * sum(p * p for p in model.parameters())

    correct = sum((y > 0) == (score.data > 0) for y, score in zip(ys, scores))
    return data_loss + reg_loss, correct / len(ys)