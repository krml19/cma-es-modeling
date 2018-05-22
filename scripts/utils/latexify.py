from collections import namedtuple

import pandas as pd
from functools import reduce

from scripts.utils.aggregations import Aggragator

eol = '\n'
hline = '\hline'
line_end = '\\\\'
row_end = line_end + ' ' + hline
sep = ' & '
pm = '\\,$\\pm$\\,'


def mappings(value):
    _mappings = {'f_2n': '2n',
                'f_2n2': '2n^2',
                'f_n3': '2n^3',
                'f_2pn': '2^n'}
    if value in _mappings.keys():
        return _mappings[value]
    else:
        return value


def math(value):
    return '${{{}}}$'.format(value)


def boldmath(value):
    return '$\\bm{{{}}}$'.format(value)


def convert_attribute_value(value):
    return boldmath(mappings(value))


def bold(text):
    return '\\textbf{{{input}}}'.format(input=text)


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

    def beautify_name(self, name: str):
        components = name.split('_')
        return '${name}_{{{n}}}^{{{k}}}$'.format(name=components[0], n=components[1], k=components[2]) if len(components) == 3 else name


    def create_row(self, series: pd.Series):
        unstucked = series.unstack().apply(
            lambda row: self.map_to_cell(row['rank_norm'], row['f_mean'], row['f_sem']))
        return unstucked

    def map_to_color(self, x: float) -> str:
        return '{},{},{}'.format(1 - x, x, 0)

    def map_to_cell(self, norm_rank: float, value: float, error: float) -> str:
        return '\cellcolor[rgb]{{{color}}} {value} {pm} {error}'.format(color=self.map_to_color(norm_rank), value=value, pm=pm, error=error)

    def header(self):
        return bold('problem') + sep + reduce(lambda x, y: convert_attribute_value(x) + sep + convert_attribute_value(y), self.attribute_values)

    def top_caption(self, name, cols):
        return '\\hline\n \multicolumn{{{count}}}{{{alignment}}}{{{name}}}'.format(count=cols, alignment='|c|', name=bold(name))

    def concat_row(self, x: pd.Series):
        return self.beautify_name(x.name) + sep + reduce(lambda xi, yi: xi + sep + yi, x)

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

        return reduced


class Component:
    value = None
    _template: str = None

    def build(self) -> str:
        return self._template.format(self.value) if self.value is not None else ''


class Label(Component):
    _template = '\\label{{{}}}\n'


class Caption(Component):
    _template = '\\caption{{{}}}\n'


class Brackets(Component):
    _template = '[{}]'


class Curly(Component):
    _template = '{{{}}}'


class Environment:
    def __init__(self):
        self._body: [str, type] = ''
        self.component_type: [str, None] = None
        self.bracket_options: Brackets = Brackets()
        self.curly_options: Curly = Curly()

    @property
    def body(self) -> [str, type]:
        return self._body

    @body.setter
    def body(self, value):
        self._body = value

    def build(self) -> str:
        bracket_options = self.bracket_options.build()
        curly_options = self.curly_options.build()
        body = self.body.build() if isinstance(self.body, Environment) else self.body
        return '\\begin{{{component}}}{curly}{brackets}\n' \
               '{body}\n' \
               '\\end{{{component}}}\n'.format(component=self.component_type, curly=curly_options, brackets=bracket_options, body=body)


class Centering(Environment):
    def __init__(self):
        super().__init__()
        self.component_type = 'centering'


class Table(Environment):
    def __init__(self):
        super().__init__()
        self.component_type = 'table'

    @Environment.body.setter
    def body(self, value):
        value = value.build() if isinstance(value, Environment) else value
        value = value + self.label.build() + self.caption.build()
        Environment.body.fset(self, value)

    label = Label()
    caption = Caption()


class Tabular(Environment):
    def __init__(self):
        super().__init__()
        self.component_type = 'tabular'

    @Environment.body.setter
    def body(self, value):
        if isinstance(value, DataTable):
            columns = ['c|'] * value.cols
            columns = '|l|' + reduce(lambda x, y: x + y, columns)
            self.curly_options.value = columns
            Environment.body.fset(self, value.create_table())
        else:
            Environment.body.fset(self, value)


Experiment = namedtuple('Experiment', ['experiment', 'benchmark_mode', 'attribute', 'caption', 'label', 'header'])


def table(experiment: Experiment):
    aggregator = Aggragator(experiment=experiment.experiment, benchmark_mode=experiment.benchmark_mode, attribute=experiment.attribute)
    data_frame = aggregator.transform()

    data_table = DataTable(data_frame, experiment.header, attribute=aggregator.attribute,
                           attribute_values=aggregator.attribute_values)

    tabular = Tabular()
    tabular.body = data_table

    table = Table()
    table.label.value = experiment.label
    table.caption.value = experiment.caption
    table.body = tabular

    centering = Centering()
    centering.body = table
    print(centering.build())


experiment1 = Experiment(experiment=1, benchmark_mode=False, attribute='standardized', caption='Wpływ standaryzacji', label='experiment1', header='Standaryzacja')
# experiment2 = Experiment(experiment=2, benchmark_mode=False, attribute='standardized', caption='', label='')
# experiment3 = Experiment(experiment=3, benchmark_mode=False, attribute='standardized', caption='', label='')
# experiment4 = Experiment(experiment=4, benchmark_mode=False, attribute='standardized', caption='', label='')
# experiment5 = Experiment(experiment=5, benchmark_mode=False, attribute='standardized', caption='', label='')
table(experiment=experiment1)
