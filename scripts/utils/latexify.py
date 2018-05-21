import pandas as pd
from functools import reduce

eol = '\n'
hline = '\hline'
line_end = '\\\\'
row_end = line_end + ' ' + hline


def bold(text):
    return '\\textbf{' + text + '}'


def row(text):
    return text + row_end + eol


class Table:

    def __init__(self, data: pd.DataFrame, top_header_name: str):
        self.cols = int(data.shape[1] / 3)

        self.t_header = self.start(data)
        self.t_end = self.end()
        self.elements = self.create_table(data, top_header_name)

    @staticmethod
    def start(data: pd.DataFrame):
        columns = ['c'] * data.shape[1]
        columns = '|l|' + reduce(lambda x, y: x + '|' + y, columns)
        return "\\begin{tabular}{" + columns + "}\n"

    @staticmethod
    def end():
        return "\n\\end{tabular}"

    def create_row_section(self, df: pd.DataFrame):
        df['row'] = df.apply(self.create_row, axis=1)

    def create_row(self, series: pd.Series):
        items = [self.map_item(series, item) for item in range(0, self.cols)]
        result = series.name + ' & ' + reduce(lambda x, y: '{} & {}'.format(x, y), items)
        return result

    def map_to_color(self, x: float) -> str:
        return '{},{},{}'.format(1 - x, x, 0)

    def cell_color(self, x: float) -> str:
        return '\cellcolor[rgb]{' + self.map_to_color(x) + '}'

    def map_item(self, series: pd.Series, item_name):
        return self.cell_color(series['norm_rank_{}'.format(item_name)]) + str(series[item_name])

    def header(self, data):
        return bold('problem') + ' & ' + reduce(lambda x, y: bold(str(x)) + ' & ' + bold(str(y)), data.columns[range(0, self.cols)])

    def top_caption(self, name, cols):
        return '\multicolumn{' + str(cols) + '}{|c|}{' + bold(name) + '}'

    def rank(self, data: pd.DataFrame):
        cols = list(map(lambda x: 'rank_{}'.format(x), data.columns[range(0, self.cols)]))
        ranks = [data[col].mean() for col in cols]
        result = bold("rank") + ' & ' + reduce(lambda x, y: bold(str(round(x, 5))) + ' & ' + bold(str(round(y, 5))), ranks)
        return result

    def create_table(self, data: pd.DataFrame, top_header_name) -> str:
        data = data.round(decimals=5)
        self.create_row_section(data)

        elements = list()
        elements.append(self.top_caption(top_header_name, self.cols + 1))
        elements.append(self.header(data))
        table_data = list(data['row'].values)
        elements.extend(table_data)
        elements.append(self.rank(data))
        elements = list(map(lambda x: row(x), elements))
        reduced: str = reduce(lambda x, y: x + y, elements)
        reduced = reduced.replace('_', '\_')

        return reduced

    def table(self) -> str:
        return self.t_header + hline + eol + self.elements + self.t_end

