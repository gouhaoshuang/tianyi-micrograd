
def draw_dot(root):
    from graphviz import Digraph

    nodes, edges = trace(root)
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    for node in sorted(nodes, key=id):
        uid = str(id(node))
        label = f"{{data {node.data:.4f} | grad {node.grad:.4f}}}"
        dot.node(uid, label=label, shape="record")
        if node._op:
            op_uid = f"op_{uid}"
            dot.node(op_uid, label=node._op)
            dot.edge(op_uid, uid)

    for parent, node in sorted(edges, key=lambda edge: (id(edge[0]), id(edge[1]))):
        dot.edge(str(id(parent)), f"op_{id(node)}")

    return dot

def trace(root):
    nodes, edges = set(), set()

    def build(node):
        if node in nodes:
            return
        nodes.add(node)
        for parent in node._prev:
            edges.add((parent, node))
            build(parent)

    build(root)
    return nodes, edges



def format_graph(root):
    nodes, _ = trace(root)
    lines = []
    for node in sorted(nodes, key=id):
        inputs = ", ".join(str(id(p)) for p in sorted(node._prev, key=id))
        lines.append(
            f"{id(node)}: data={node.data:.4f} grad={node.grad:.4f} "
            f"op={node._op or 'leaf'} inputs=[{inputs}]"
        )
    return "\n".join(lines)

