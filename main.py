import ast

import graphviz as gv
import numbers
import re
from uuid import uuid4 as uuid

'''
Утилита для визуализации AST
'''

def main(code, label=None, name='graph', folder = 'unknown'):
    code_ast = ast.parse(code)
    transformed_ast = transform_ast(code_ast)

    renderer = GraphRenderer()
    renderer.render(transformed_ast, label=label, name=name, folder = folder)


def transform_ast(code_ast):
    if isinstance(code_ast, ast.AST):
        node = {to_camelcase(k): transform_ast(getattr(code_ast, k)) for k in code_ast._fields}
        node['node_type'] = to_camelcase(code_ast.__class__.__name__)
        return node
    elif isinstance(code_ast, list):
        return [transform_ast(el) for el in code_ast]
    else:
        return code_ast


def to_camelcase(string):
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()


class GraphRenderer:

    graphattrs = {
        'labelloc': 't',
        'fontcolor': 'white',
        'bgcolor': '#333333',
        'margin': '0',
    }

    nodeattrs = {
        'color': 'white',
        'fontcolor': 'white',
        'style': 'filled',
        'fillcolor': '#006699',
    }

    edgeattrs = {
        'color': 'white',
        'fontcolor': 'white',
    }

    _graph = None
    _rendered_nodes = None

    @staticmethod
    def _escape_dot_label(str):
        return str.replace("\\", "\\\\").replace("|", "\\|").replace("<", "\\<").replace(">", "\\>")

    def _render_node(self, node):
        if isinstance(node, (str, numbers.Number)) or node is None:
            node_id = uuid()
        else:
            node_id = id(node)
        node_id = str(node_id)

        if node_id not in self._rendered_nodes:
            self._rendered_nodes.add(node_id)
            if isinstance(node, dict):
                self._render_dict(node, node_id)
            elif isinstance(node, list):
                self._render_list(node, node_id)
            else:
                self._graph.node(node_id, label=self._escape_dot_label(str(node)))

        return node_id

    def _render_dict(self, node, node_id):
        self._graph.node(node_id, label=node.get("node_type", "[dict]"))
        for key, value in node.items():
            if key == "node_type":
                continue
            child_node_id = self._render_node(value)
            self._graph.edge(node_id, child_node_id, label=self._escape_dot_label(key))

    def _render_list(self, node, node_id):
        self._graph.node(node_id, label="[list]")
        for idx, value in enumerate(node):
            child_node_id = self._render_node(value)
            self._graph.edge(node_id, child_node_id, label=self._escape_dot_label(str(idx)))

    def render(self, data, *, label=None, name='graph',folder = 'unknown'):
        graphattrs = self.graphattrs.copy()
        if label is not None:
            graphattrs['label'] = self._escape_dot_label(label)
        graph = gv.Digraph(name, graph_attr=graphattrs, node_attr=self.nodeattrs, edge_attr=self.edgeattrs)

        self._graph = graph
        self._rendered_nodes = set()
        self._render_node(data)
        self._graph = None
        self._rendered_nodes = None

        graph.format = "pdf"
        graph.render(directory=f'graphviz/{folder}')


if __name__ == '__main__':
    fname = 'arima.py'
    f = open(f'plagiat/files/{fname}')
    src = f.read()
    main(src, name=f'files_{fname}', folder=fname)
    tree = ast.parse(source=src)
    print(ast.dump(tree, indent=4))



    f1 = open(f'plagiat/plagiat1/{fname}')
    src1 = f1.read()
    main(src1, name=f'plagiat1_{fname}', folder=fname)
    tree1 = ast.parse(source=src1)
    print(ast.dump(tree1, indent=4))

    f2 = open(f'plagiat/plagiat2/{fname}')
    src2 = f2.read()
    main(src2, name=f'plagiat2_{fname}', folder=fname)
