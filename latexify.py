import pandas as pd
from functools import reduce
from aggregations import Aggragator
from file_helper import write_tex_table
import sys
import numpy as np
from logger import Logger

log = Logger(name='latex')

eol = '\n'
line_end = '\\\\'
row_end = line_end
sep = ' & '
pm = '\\,$\\pm$\\,'


def mappings(value):
    if isinstance(value, str):
        return value \
            .replace('f_2np2', '2n^2') \
            .replace('f_2n', '2n') \
            .replace('f_n3', 'n^3') \
            .replace('f_2pn', '2^n') \
            .replace('tp', 'TP') \
            .replace('fp', 'FP') \
            .replace('tn', 'TN') \
            .replace('fn', 'FN') \
            .replace('accuracy', 'acc') \
            .replace('precision', 'p') \
            .replace('recall', 'r')
    else:
        return value


def header_mappings(value):
    if isinstance(value, str):
        return value.replace('standardized', 'Standaryzacja') \
            .replace('margin', 'Margines') \
            .replace('constraints_generator', 'Ograniczenia') \
            .replace('sigma', '\sigma') \
            .replace('clustering', 'k_{min}') \
            .replace('total_experiments', 'Total')
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


def bm(text):
    if '$' in text:
        return '$\\bm{{{}}}$'.format(text)
    else:
        return '\\textbf{{{input}}}'.format(input=text)


class Formatter:
    @staticmethod
    def format_color(x: float) -> str:
        x = np.round(x, decimals=5)
        return '{},{},{}'.format(1 - x, x, 0)

    @staticmethod
    def format_error(x: float) -> str:
        return '\\begin{tikzpicture}[y=0.75em,baseline=1pt]\\draw[very thick] (0,0) -- (0,%f);\\end{tikzpicture}' % x

    @staticmethod
    def format_cell(norm_rank: float, value: float, error: float) -> str:
        return '\cellcolor[rgb]{%s} %0.2f %s ' % \
               (Formatter.format_color(norm_rank), value, Formatter.format_error(error))

    @staticmethod
    def format_header(attribute_values, name):
        attributes = list(map(lambda x: convert_attribute_value(x), attribute_values))
        header = bm(name) + sep + reduce(lambda x, y: x + sep + y, attributes)
        return header

    @staticmethod
    def first_row_formatter(value):
        return Formatter.format_model_name(value)

    @staticmethod
    def format_rank(series: pd.Series):
        return bold(series.name) + sep + reduce(lambda x, y: str(x) + sep + str(y), series.apply(lambda x: "%0.2f" % x))

    @staticmethod
    def format_model_name(name):
        if isinstance(name, tuple):
            components = list(name)
        elif isinstance(name, list):
            components = name
        else:
            components = name.split('_')
        formats = ['%s', '^{%s}', '_{%s}']
        items = [format % component for format, component in zip(formats, components)]

        return '$%s$' % reduce(lambda x, y: x + y, items).title()


class DataTable:
    decimals = 2

    def __init__(self, data: pd.DataFrame, top_header_name: str, attribute: str, attribute_values: list):
        self.cols = len(attribute_values)
        self.attribute = attribute
        self.attribute_values = attribute_values
        self.top_header_name = top_header_name
        self.data = data
        self.title: str = ''

    def create_row_section(self, df: pd.DataFrame):
        keys = [('row', key) for key in self.attribute_values]
        df[keys] = df.apply(self.format_series_element, axis=1)

    def format_series_element(self, series: pd.Series):
        unstucked = series.unstack().apply(
            lambda row: Formatter.format_cell(row['rank_norm'], row['f_mean'], row['sem_norm']))
        return unstucked

    def concat_row(self, x: pd.Series):
        component = '\\midrule\n' if x.name.split('_')[2] == '2' else ''
        return component + Formatter.format_model_name(x.name) + sep + reduce(lambda xi, yi: xi + sep + yi,
                                                                              x) + row_end + eol

    def rank(self, data: pd.DataFrame):
        result = '\\midrule\n' + bold("rank") + sep + reduce(lambda x, y: str(x) + sep + str(y),
                                                             map(lambda x: "%0.2f" % x,
                                                                 data['rank'].mean())) + row_end + eol
        return result

    def build(self) -> str:
        data = self.data.round(decimals=self.decimals)
        self.create_row_section(data)

        table_data = data['row'].apply(self.concat_row, axis=1)
        elements = list(table_data)
        elements.append(self.rank(data))
        header = Formatter.format_header(self.attribute_values, self.title)
        reduced: str = reduce(lambda x, y: x + y, elements)

        return '\\toprule\n ' \
               '\multicolumn{{{count}}}{{{alignment}}}{{{name}}} \\\\ \n' \
               '\\midrule\n' \
               '{header} \\\\ \n' \
               '{body}' \
               '\\bottomrule\n'.format(count=self.cols + 1, alignment='c', name=self.top_header_name, header=header,
                                       body=reduced)


class DataPivotTable(DataTable):
    def __init__(self, data: pd.DataFrame, top_header_name: str, attribute: str, attribute_values: list,
                 pivot: bool = True, header_formatter=Formatter.format_header, row_formatter=Formatter.first_row_formatter):
        super().__init__(data=data, top_header_name=top_header_name, attribute=attribute,
                         attribute_values=attribute_values)
        self.data: pd.DataFrame = self.data.T
        self.pivot = pivot
        self.formatters = {'header': header_formatter,
                           'first_row': row_formatter}

    def format_series(self, series):
        col_formatter = lambda s, attribute: Formatter.format_cell(s[('rank_norm', attribute)],
                                                                   s[('f_mean', attribute)],
                                                                   s[('sem_norm', attribute)])
        formatted_cols = [col_formatter(series, attribute) for attribute in self.attribute_values]
        return pd.Series(data=formatted_cols, name=series.name)

    def format_row(self, series: pd.Series):
        return self.formatters['first_row'](series.name) + sep + reduce(lambda x, y: x + sep + y, series)

    def rank(self, data: pd.DataFrame):
        return pd.Series(data=data.loc['rank'].mean(axis=1), name='rank')

    def build(self) -> str:
        header = self.formatters['header'](self.attribute_values, self.title)
        reducer = lambda x, y: str(x) + row_end + eol + str(y)
        rank = Formatter.format_rank(self.rank(self.data))

        body = self.data.apply(self.format_series)
        body = body.T if self.pivot else body
        body = body.apply(self.format_row, axis=1)
        body = reduce(reducer, body.values)

        return '\\toprule \n' \
               '\\multicolumn{{{count}}}{{{alignment}}}{{{name}}} \\\\ \n' \
               '\\midrule \n' \
               '{header} \\\\ \n' \
               '\\midrule \n' \
               '{body} \\\\ \n' \
               '\\midrule \n' \
               '{rank} \\\\ \n' \
               '\\bottomrule \n'.format(count=self.cols + 1, alignment='c', name=self.top_header_name,
                                       header=header, body=body, rank=rank)


class InfoTable:
    def __init__(self, info: dict):
        self.info = info

    def build(self) -> str:
        reducer = lambda x, y: str(x) + sep + str(y)
        keys = list(map(lambda x: boldmath(header_mappings(x)), self.info.keys()))
        values = list(map(lambda x: math(mappings(x)), self.info.values()))
        header = reduce(reducer, keys)
        body = reduce(reducer, values)

        return '\\toprule\n' \
               '{header} \\\\\n' \
               '\\midrule\n' \
               '{body} \\\\\n' \
               '\\bottomrule\n'.format(header=header, body=body)


class ConfusionMatrix:
    def __init__(self, cm: pd.DataFrame):
        self.cm: pd.DataFrame = cm
        self.cols = len(cm.keys())
        pass

    @property
    def header(self):
        attributes = list(map(lambda x: convert_attribute_value(x), self.cm.keys()))
        return bold('') + sep + reduce(lambda x, y: x + sep + y, attributes)

    @property
    def body(self):
        formatter = lambda x: "%0.3f" % x
        map_series = lambda s: str(s.name) + sep + reduce(lambda xi, yi: xi + sep + yi, s) + row_end
        reducer = lambda r1, r2: r1 + eol + r2

        body = self.cm.applymap(formatter)
        body = body.apply(map_series, axis=1)
        body = reduce(reducer, body)
        return body

    def build(self) -> str:
        return '\\toprule\n' \
               '{header} \\\\\n' \
               '\\midrule\n' \
               '{body} \n' \
               '\\bottomrule\n'.format(header=self.header, body=self.body)


class Component:
    value = None
    _template: str = None

    def __init__(self, value=None):
        self.value = value

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


class Comment(Component):
    _template = '% {0: <20}\n'


class Environment:
    def __init__(self):
        self._body: [str, type, dict] = ''
        self.component_type: [str, None] = None
        self.comment: CommentBlock = None
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
        comment = self.comment.build() if isinstance(self.comment, CommentBlock) else ''
        return '{comment}\n' \
               '\\begin{{{component}}}{curly}{brackets}\n' \
               '{body}\n' \
               '\\end{{{component}}}\n'.format(component=self.component_type, curly=curly_options,
                                               brackets=bracket_options, body=body, comment=comment)


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
        if isinstance(value, DataTable) or isinstance(value, ConfusionMatrix) or isinstance(value, DataPivotTable):
            columns = 'l' + 'r' * value.cols
            self.curly_options.value = columns
            Environment.body.fset(self, value.build())
        elif isinstance(value, InfoTable):
            columns = 'c' * len(value.info)
            self.curly_options.value = columns
            Environment.body.fset(self, value.build())
        else:
            Environment.body.fset(self, value)


class CommentBlock:
    body = None

    def __init__(self, body):
        self.body = body

    def build(self):
        if isinstance(self.body, dict):
            items: dict = self.body
            mapped_items = list(
                map(lambda item: Comment(value="%25s\t\t%s" % (item[0], item[1])).build(), items.items()))
            return reduce(lambda x, y: x + y, mapped_items)
        return ''


class Experiment:
    def __init__(self, index: int, attribute, header, table, split, benchmark_mode=False):
        self.index = index
        self.attribute = attribute
        self.header = header
        self.table = table
        self.split = split
        self.benchmark_mode = benchmark_mode

    def __str__(self):
        return "Experiment %d:{%s} " % (self.index, self.attribute)


def table(experiment: Experiment):
    aggregator = Aggragator(experiment=experiment.index, benchmark_mode=experiment.benchmark_mode,
                            attribute=experiment.attribute)

    data_frame = aggregator.transform(split=experiment.split)
    if isinstance(data_frame, list):
        for data in data_frame:
            title = bm('$n \\backslash |X|$')
            experiment.header = Formatter.format_model_name(data[1])
            save(experiment=experiment, aggregator=aggregator, data_frame=data[0], filename=
            'experiment%d-%s-%d' % (experiment.index, data[1][0], data[1][1]), title=title)
    else:
        save(experiment=experiment, aggregator=aggregator, data_frame=data_frame,
             filename='experiment%d' % experiment.index)


def save(experiment: Experiment, aggregator: Aggragator, data_frame: pd.DataFrame, title='problem',
         filename='experiment'):
    log.info("Start: %s" % experiment)
    data_table = experiment.table(data_frame, experiment.header, attribute=aggregator.attribute,
                                  attribute_values=aggregator.attribute_values)
    data_table.title = title

    tabular = Tabular()
    tabular.body = data_table
    tabular.comment = CommentBlock(aggregator.info)

    log.info("Writing: %s" % experiment)
    if len(sys.argv) > 1:
        write_tex_table(filename=filename, data=tabular.build(), path=sys.argv[1])
    log.info("Finished: %s" % experiment)


if __name__ == '__main__':
    experiment1 = Experiment(index=1, attribute='standardized', header=bold('Standaryzacja'), table=DataPivotTable,
                             split=None)
    experiment2 = Experiment(index=2, attribute='constraints_generator', header=bold('Liczba ograniczeń'),
                             table=DataPivotTable, split=None)
    experiment3 = Experiment(index=3, attribute='clustering', header=boldmath('k_{min}'),
                             table=DataPivotTable, split=None)
    experiment4 = Experiment(index=4, attribute='sigma', header=boldmath('\sigma'),
                             table=DataPivotTable, split=None)
    experiment5 = Experiment(index=5, attribute='margin', header=bold('Margines'),
                             table=DataPivotTable, split=None)
    experiment6 = Experiment(index=6, attribute='train_sample', header=bold('|X|'),
                             table=DataPivotTable, split=['name', 'k'])

    for experiment in [experiment1, experiment2, experiment3, experiment4, experiment5, experiment6]:
    # for experiment in [experiment6]:
    # for experiment in [experiment1]:
        table(experiment=experiment)
