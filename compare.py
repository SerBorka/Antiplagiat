import ast
import numpy as np
import argparse

def file_to_tree(str):
    f = open(str, encoding='latin-1')
    src = f.read()
    return ast.parse(source=src)


class PyAnalizer:

    def __init__(self, str):
        self.tree = file_to_tree(str)
        self.docstrings = {}
        self.codelist = []
        self.dict_borders = {}
        self.deldocs()
        self.rebildtree()
        self.code_borders = [set() for i in range(len(self.codelist))]
        self.chekborders()
        self.collect_code_borders()

    def rec(self, node, nodefunc):
        nodefunc(node)
        for n in ast.iter_child_nodes(node):
            self.rec(n, nodefunc)

    def deldocs(self):
        self.rec(self.tree, self._deletedocstring)

    def rebildtree(self):
        ast.fix_missing_locations(self.tree)
        cleancode = ast.unparse(self.tree)
        # print(cleancode)
        self.codelist = cleancode.split('\n')
        # for i, line in enumerate(self.codelist):
        #    print(i + 1, line)
        self.tree = ast.parse(cleancode)

    def chekborders(self):
        self.rec(self.tree, self._collect_positions)

    def _deletedocstring(self, node: ast.AST):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            blist = node.body
            new_body = []
            node_name = type(node).__name__
            if node_name == 'Module':
                nname = 'File'
            else:
                nname = node.name

            for i in range(len(blist)):
                if isinstance(blist[i], ast.Expr) and isinstance(blist[i].value, ast.Constant):
                    exp = blist[i]
                    if node_name not in self.docstrings:
                        self.docstrings[node_name] = [[nname, exp.value.value]]
                    else:
                        self.docstrings[node_name].append([nname, exp.value.value])
                else:
                    new_body.append(blist[i])
            if len(new_body) == 0:
                new_body.append(ast.Pass())
            node.body = new_body

    def _collect_positions(self, node):
        node_name = type(node).__name__
        try:
            if node_name not in self.dict_borders:
                self.dict_borders[node_name] = [[node.lineno, node.end_lineno]]
            else:
                self.dict_borders[node_name].append([node.lineno, node.end_lineno])
        except:
            AttributeError

    def collect_code_borders(self):
        b_dict = self.dict_borders
        for key in b_dict:
            b_list = b_dict[key]
            for ind1, ind2 in b_list:
                for i in range(ind1 - 1, ind2):
                    self.code_borders[i].add(key)

    def print_code(self):
        for i, line in enumerate(self.codelist):
            print(i, line)


def lowen_dist(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current_row = range(n + 1)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)
    return current_row[n]


class FilesComparing:
    typelist = ['AnnAssign', 'Assign', 'Attribute', 'AugAssign', 'BinOp', 'BoolOp', 'Call', 'ClassDef', 'Compare',
                'Constant', 'Continue', 'Dict', 'DictComp', 'ExceptHandler', 'Expr', 'For', 'FormattedValue',
                'FunctionDef', 'If', 'IfExp', 'Import', 'ImportFrom', 'JoinedStr', 'Lambda', 'List', 'ListComp',
                'Name', 'Raise', 'Return', 'Set', 'SetComp', 'Slice', 'Starred', 'Subscript', 'Try', 'Tuple',
                'UnaryOp', 'With', 'arg', 'keyword']

    def __init__(self, file1_path, file2_path):
        self.file1 = PyAnalizer(file1_path)
        self.file2 = PyAnalizer(file2_path)
        self.dict_comp = {}
        self.simple_value = 0
        self.compare_files()
        self.compare_vector = [0] * len(self.typelist)
        self.fill_compare_vector()

        #print(self.compare_vector)
        #self.file1.print_code()
        #self.file2.print_code()

    def compare_line_code(self, line, code):
        ld = lowen_dist(line, code)
        diff = len(code) - len(line)
        return 1 - (ld - diff) / len(line)

    def compare_files(self):

        for i, types in enumerate(self.file1.code_borders):
            line = self.file1.codelist[i]
            if len(types) == 0 or len(line) == 0:
                continue

            imax, maxcoef = 0, 0
            for ind, codeline in enumerate(self.file2.codelist):
                coef = self.compare_line_code(line, codeline)
                # print(ind,coef)
                if coef > maxcoef:
                    imax, maxcoef = ind, coef
            #print(i, 'looks like', imax, maxcoef)
            for t in types.intersection(self.file2.code_borders[imax]):
                if t not in self.dict_comp:
                    self.dict_comp[t] = [maxcoef]
                else:
                    self.dict_comp[t].append(maxcoef)
        for key in self.dict_comp:
            self.dict_comp[key] = np.mean(self.dict_comp[key])

    def fill_compare_vector(self):
        score = []
        for i, key in enumerate(self.typelist):
            if key in self.dict_comp:
                self.compare_vector[i] = self.dict_comp[key]
                score.append(self.dict_comp[key]**2)
        self.simple_value = np.mean(score)




if __name__ == '__main__':

    pars = argparse.ArgumentParser(description='')
    pars.add_argument('input', type=str, help='Input file with a list of file pairs to check.')
    pars.add_argument('scores', type=str, help='Program text similarity evaluation file.')
    args = pars.parse_args()
    scores = args.scores
    inp = args.input
    f = open(inp)
    #python compare.py input.txt scores.txt
    for line in f:
        files = line.split()
        file1 = files[0]
        file2 = files[1]
        fc = FilesComparing(file1,file2)
        with open(scores, 'a') as scr:
            scr.write(str(fc.simple_value)+'\n')




