from collections import namedtuple
import pandas as pd
from functools import reduce
from aggregations import Aggragator
from file_helper import write_tex_table
import sys
import logging

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


def row(text):
    return text + row_end + eol


class Formatter:
    @staticmethod
    def format_color(x: float) -> str:
        return '{},{},{}'.format(1 - x, x, 0)

    @staticmethod
    def format_error(x: float) -> str:
        return '\\begin{tikzpicture}[y=0.75em,baseline=1pt]\\draw[very thick] (0,0) -- (0,%f);\\end{tikzpicture}' % x

    @staticmethod
    def format_cell(norm_rank: float, value: float, error: float) -> str:
        return '\cellcolor[rgb]{%s} %0.2f %s ' % \
               (Formatter.format_color(norm_rank), value, Formatter.format_error(error))

    @staticmethod
    def format_header(attribute_values, cols, name):
        attributes = list(map(lambda x: convert_attribute_value(x), attribute_values))
        header = bold('problem') + sep + reduce(lambda x, y: x + sep + y, attributes)
        return '\\toprule\n ' \
               '\multicolumn{{{count}}}{{{alignment}}}{{{name}}} \\\\ \n' \
               '\\midrule\n' \
               '{header} \\\\ \n'.format(count=cols, alignment='c', name=name, header=header)

    @staticmethod
    def format_model_name(name):
        components = name.split('_')
        return '${name}_{{{n}}}^{{{k}}}$'.format(name=components[0], n=components[2], k=components[1]) if len(
            components) == 3 else name


class DataTable:
    decimals = 2

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
            lambda row: Formatter.format_cell(row['rank_norm'], row['f_mean'], row['sem_norm']))
        return unstucked

    def concat_row(self, x: pd.Series):
        component = '\\midrule\n' if x.name.split('_')[2] == '2' else ''
        return component + Formatter.format_model_name(x.name) + sep + reduce(lambda xi, yi: xi + sep + yi, x) + row_end + eol

    def rank(self, data: pd.DataFrame):
        result = '\\midrule\n' + bold("rank") + sep + reduce(lambda x, y: str(x) + sep + str(y),
                                                             map(lambda x: "%0.2f" % x, data['rank'].mean())) + row_end + eol
        return result

    def create_table(self) -> str:
        data = self.data.round(decimals=self.decimals)
        self.create_row_section(data)

        table_data = data['row'].apply(self.concat_row, axis=1)
        elements = list(table_data)
        elements.append(self.rank(data))
        header = Formatter.format_header(self.attribute_values, self.cols + 1, self.top_header_name)
        reduced: str = header + reduce(lambda x, y: x + y, elements) + '\\bottomrule\n'

        return reduced


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
        if isinstance(value, DataTable):
            columns = ['r'] * value.cols
            columns = 'l' + reduce(lambda x, y: x + y, columns)
            self.curly_options.value = columns
            Environment.body.fset(self, value.create_table())
        elif isinstance(value, InfoTable):
            columns = 'c' * len(value.info)
            self.curly_options.value = columns
            Environment.body.fset(self, value.build())
        elif isinstance(value, ConfusionMatrix):
            columns = 'l' + 'r' * value.cols
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


Experiment = namedtuple('Experiment', ['experiment', 'benchmark_mode', 'attribute', 'header'])


def table(experiment: Experiment):
    aggregator = Aggragator(experiment=experiment.experiment, benchmark_mode=experiment.benchmark_mode,
                            attribute=experiment.attribute)
    data_frame = aggregator.transform()

    data_table = DataTable(data_frame, experiment.header, attribute=aggregator.attribute,
                           attribute_values=aggregator.attribute_values)

    tabular = Tabular()
    tabular.body = data_table
    tabular.comment = CommentBlock(aggregator.info)

    document = tabular.build()

    print(document)
    # cm = ConfusionMatrix(aggregator.cm)
    # tabular2 = Tabular()
    # tabular2.body = cm
    # cm_info = tabular2.build()
    # print(cm_info)

    if len(sys.argv) > 1:
        write_tex_table(filename='experiment%d' % experiment.experiment, data=document, path=sys.argv[1])
        # write_tex_table(filename='cm%d' % experiment.experiment, data=cm_info, path=sys.argv[1])


if __name__ == '__main__':
    experiment1 = Experiment(experiment=1, benchmark_mode=False, attribute='standardized', header=bold('Standaryzacja'))
    experiment2 = Experiment(experiment=2, benchmark_mode=False, attribute='constraints_generator',
                             header=bold('Liczba ograniczeń'))
    experiment3 = Experiment(experiment=3, benchmark_mode=False, attribute='clustering', header=boldmath('k_{min}'))
    experiment4 = Experiment(experiment=4, benchmark_mode=False, attribute='sigma', header=boldmath('\sigma'))
    experiment5 = Experiment(experiment=5, benchmark_mode=False, attribute='margin', header=bold('Margines'))
    experiment6 = Experiment(experiment=6, benchmark_mode=False, attribute='train_sample', header=bold('|X|'))

    for experiment in [experiment1, experiment2, experiment3, experiment4, experiment5]:
        table(experiment=experiment)
