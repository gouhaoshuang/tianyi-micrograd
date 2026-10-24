from tianyi_micrograd.nn import Neuron


DATA = [(-1.0, -1.0), (0.0, 1.0), (1.0, 3.0)]

def build_model():
    model = Neuron(1, nonlin=False)
    model.w[0].data = 0.0
    model.b.data = 0.0
    return model

def mean_squared_error(model, data):
    errors = [(model([x]) - y)**2 for x, y in data]
    return sum(errors) / len(errors)


def train(steps=100, learning_rate=0.1):
    model = build_model()
    history = []

    for step in range(steps):
        loss = mean_squared_error(model, DATA)
        history.append(loss.data)

        model.zero_grad()
        loss.backward()
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

    history.append(mean_squared_error(model, DATA).data)
    return model, history


def main():
    model, history = train()
    print(f"initial loss: {history[0]:.6e}")
    print(f"final loss:   {history[-1]:.6e}")
    print(f"w={model.w[0].data:.6f}, b={model.b.data:.6f}")
    for x, target in DATA:
        prediction = model([x]).data
        print(f"x={x:+.1f}, target={target:+.1f}, prediction={prediction:+.6f}")


if __name__ == "__main__":
    main()