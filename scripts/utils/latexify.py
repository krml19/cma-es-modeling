import pandas as pd
from functools import reduce

from scripts.utils.aggregations import Aggragator

eol = '\n'
hline = '\hline'
line_end = '\\\\'
row_end = line_end + ' ' + hline
sep = ' & '
pm = '\\,$\\pm$\\,'

def bold(text):
    return '\\textbf{' + text + '}'


def row(text):
    return text + row_end + eol


class DataTable:
    decimals = 5

    def __init__(self, data: pd.DataFrame, top_header_name: str, attribute: str, attribute_values: list):
        self.cols = len(attribute_values)
        self.attribute = attribute
        self.attribute_values = attribute_values
        self.top_header_name = top_header_name
        self.data = data

    def create_row_section(self, df: pd.DataFrame):
        keys = [('row', key) for key in self.attribute_values]
        df[keys] = df.apply(self.create_row, axis=1)

    def create_row(self, series: pd.Series):
        unstucked = series.unstack().apply(
            lambda row: self.map_to_cell(row['rank_norm'], row['f_mean'], row['f_sem']))
        return unstucked

    def map_to_color(self, x: float) -> str:
        return '{},{},{}'.format(1 - x, x, 0)

    def map_to_cell(self, norm_rank: float, value: float, error: float) -> str:
        return '\cellcolor[rgb]{' + self.map_to_color(norm_rank) + '} ' + str(value) + pm + str(error)

    def header(self):
        return bold('problem') + sep + reduce(lambda x, y: bold(str(x)) + sep + bold(str(y)), self.attribute_values)

    def top_caption(self, name, cols):
        return hline + eol + '\multicolumn{' + str(cols) + '}{|c|}{' + bold(name) + '}'

    def concat_row(self, x):
        return x.name + sep + reduce(lambda xi, yi: xi + sep + yi, x)

    def rank(self, data: pd.DataFrame):
        result = bold("rank") + sep + reduce(lambda x, y: str(x) + sep + str(y), data['rank'].mean().round(decimals=self.decimals))
        return result

    def create_table(self) -> str:
        data = self.data.round(decimals=self.decimals)
        self.create_row_section(data)

        elements = list()
        elements.append(self.top_caption(self.top_header_name, self.cols + 1))
        elements.append(self.header())
        table_data = data['row'].apply(self.concat_row, axis=1)
        elements.extend(list(table_data))
        elements.append(self.rank(data))
        elements = list(map(lambda x: row(x), elements))
        reduced: str = reduce(lambda x, y: x + y, elements)
        reduced = reduced.replace('_', '\_')

        return reduced


class Component:
    _body: [str, type] = ''
    component_type: [str, None] = None
    bracket_options: [str, None] = None
    curly_options: [str, None] = None

    @property
    def body(self) -> [str, type]:
        return self._body

    @body.setter
    def body(self, value):
        self._body = value

    def build(self) -> str:
        bracket_options = '[{}]'.format(self.bracket_options) if self.bracket_options is not None else ''
        curly_options = '{{{}}}'.format(self.curly_options) if self.curly_options is not None else ''
        body = self.body.build() if isinstance(self.body, Component) else self.body
        return '\\begin{{{component}}}{curly}{brackets}\n' \
               '{body}\n' \
               '\\end{{{component}}}\n'.format(component=self.component_type, curly=curly_options, brackets=bracket_options, body=body)


class Centering(Component):
    component_type = 'centering'


class Table(Component):
    component_type = 'table'

    @Component.body.setter
    def body(self, value):
        label = '\\label{{{label}}}\n'.format(label=self.label) if self.label is not None else ''
        caption = '\\caption{{{caption}}}'.format(caption=self.caption) if self.label is not None else ''
        value = value.build() if isinstance(value, Component) else value
        value = value + label + caption
        Component.body.fset(self, value)

    label = 'label'
    caption = 'caption'

class Tabular(Component):
    component_type = 'tabular'

    @Component.body.setter
    def body(self, value):
        if isinstance(value, DataTable):
            columns = ['c|'] * value.cols
            columns = '|l|' + reduce(lambda x, y: x + y, columns)
            self.curly_options = columns
            Component.body.fset(self, value.create_table())
        else:
            Component.body.fset(self, value)


aggregator = Aggragator()
data_frame = aggregator.fit()

data_table = DataTable(data_frame, 'Standaryzacja', attribute=aggregator.attribute, attribute_values=aggregator.attribute_values)

tabular = Tabular()
tabular.body = data_table

table = Table()
table.body = tabular

centering = Centering()
centering.body = table
print(centering.build())