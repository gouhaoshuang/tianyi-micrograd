import sys
from pathlib import Path

from tianyi_micrograd.engine import Value
from tianyi_micrograd.visualize import draw_dot, format_graph


def main(render=False):
    x = Value(3.0)
    k = Value(2.0)
    u = x * k
    loss = u * u + u
    loss.backward()
    print(format_graph(loss))

    if render:
        Path("outputs").mkdir(exist_ok=True)
        path = draw_dot(loss).render(filename="graph", directory="outputs", cleanup=True)
        print(f"saved: {path}")


if __name__ == "__main__":
    main(render="--render" in sys.argv[1:])