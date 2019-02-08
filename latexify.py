import pandas as pd
from functools import reduce
from aggregations import Aggragator, Measure, MeasureF, MeasureF1, MeasureTime
from file_helper import write_file
import sys
import numpy as np
from logger import Logger
import scipy.stats as stats
from collections import namedtuple

TableData = namedtuple('TableData',
                               'cols alignment header_name groups header body formatted_rank pvalue groups_range grouped_midrule header_midrule')

log = Logger(name='latex')

eol = '\n'
line_end = '\\\\'
row_end = line_end
sep = ' & '
pm = '\\,$\\pm$\\,'


def identify(x):
    print(x)
    return x


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
    return math(mappings(value))


def bold(text):
    return '\\textbf{{{input}}}'.format(input=text)


def bm(text):
    if '$' in text:
        return '$\\bm{{{}}}$'.format(text)
    else:
        return '\\textbf{{{input}}}'.format(input=text)


class Formatter:
    @staticmethod
    def format_color(x: float, reverse = False) -> str:
        if reverse:
            x = 1 - x
        x = np.round(x, decimals=5)
        return '{green!70!lime!%d!red!70!yellow!80!white}' % int(100 * x)

    @staticmethod
    def format_error(x: float) -> str:
        # x = min(x, 1.0) if x is float else 0.0
        # x = max(x, np.nan)
        return '\\begin{tikzpicture}[y=0.75em,baseline=1pt]\\draw[very thick] (0,0) -- (0,%f);\\end{tikzpicture}' % x

    @staticmethod
    def format_cell(norm_rank: float, value: float, error: float, reverse_colors: bool = False) -> str:
        error = error / value if value > 0 else 0
        error = min(error, 1)
        return '\cellcolor%s %0.2f %s ' % \
               (Formatter.format_color(norm_rank, reverse_colors), value, Formatter.format_error(error))

    @staticmethod
    def format_header(attribute_values, name):
        attribute_values = ["(1,1)", "(1,\infty)", "(2,\infty)"] * 2 if attribute_values == [0, 1, 2] * 2 else attribute_values
        attributes = list(map(lambda x: convert_attribute_value(x), attribute_values))
        header = math(name) + sep + reduce(lambda x, y: x + sep + y, attributes)
        return header

    @staticmethod
    def first_row_formatter(value):
        return Formatter.format_model_name(value)

    @staticmethod
    def format_rank(series: pd.Series):
        return math(series.name) + sep + reduce(lambda x, y: str(x) + sep + str(y),
                                                series.apply(lambda x: "%0.3f" % x if np.isfinite(x) else "---"))

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
        result = '\\midrule\n' + math("Rank") + sep + reduce(lambda x, y: str(x) + sep + str(y),
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
                 pivot: bool = True, header_formatter=Formatter.format_header,
                 row_formatter=Formatter.first_row_formatter, reverse_colors=False):
        data.fillna(value=0, inplace=True)
        super().__init__(data=data, top_header_name=top_header_name, attribute=attribute,
                         attribute_values=attribute_values)

        self.pivot = pivot
        self.formatters = {'header': header_formatter,
                           'first_row': row_formatter}
        self.measures = data.index.levels[0]
        self.total_cols = len(self.measures) * self.cols + 1
        self.reverse_colors = reverse_colors

    def format_series(self, series):
        col_formatter = lambda s, attribute: Formatter.format_cell(s[('rank_norm', attribute)],
                                                                   s[('f_mean', attribute)],
                                                                   s[('sem_norm', attribute)],
                                                                   self.reverse_colors)
        formatted_cols = [col_formatter(series, attribute) for attribute in self.attribute_values]
        return pd.Series(data=formatted_cols, name=series.name)

    def format_row(self, series: pd.Series):
        return self.formatters['first_row'](series.name) + sep + reduce(lambda x, y: x + sep + y, series)

    def rank(self):
        return pd.Series(data=self.data['rank'].mean(level=0).stack(), name='Rank')

    def pvalue(self):
        df = pd.DataFrame()
        for measure in self.measures:
            df[measure] = self.stats(self.data['rank'].unstack(level=1).loc[measure].unstack())
        return pd.Series(data=df.unstack(), name='p-value')

    def stats(self, df: pd.DataFrame):
        argmin = df.mean(axis=1).argmin()
        df2 = df.apply(lambda s: stats.wilcoxon(x=s, y=df.loc[argmin]).pvalue, axis=1)
        return df2

    def gr_midrules(self, add_first_col=True):
        grouped_midrules = ''
        i = 2
        for _ in self.measures:
            template = '\\cmidrule(r{10pt}){%d-%d}' if i == 2 else '\\cmidrule{%d-%d}'
            grouped_midrules = grouped_midrules + (
                        template % (i - 1 if i == 2 and add_first_col else i, i + self.cols - 1))
            i = i + self.cols
        return grouped_midrules

    def build(self) -> TableData:
        header = self.formatters['header'](self.attribute_values * len(self.measures), self.title)
        reducer = lambda x, y: str(x) + row_end + eol + str(y)
        reducer2 = lambda x, y: str(x) + str(y)

        body = self.data.apply(self.format_series, axis=1)
        body = body if self.pivot else body.T
        body = body.T.stack().swaplevel().unstack().apply(self.format_row, axis=1)
        # body = reduce(reducer2, body.values)
        body = body.values
        # groups = ['\\multicolumn{{{count}}}{{{alignment}}}{{{name}}}'.format(count=self.cols,
        #                                                                      alignment='c', name=math(name))
        #           for name in self.measures]
        # groups = sep + reduce(lambda x, y: x + sep + y, groups)
        groups = [math(name) for name in self.measures]

        rank = self.rank()
        formatted_rank = Formatter.format_rank(rank)
        pvalue = Formatter.format_rank(self.pvalue())

        grouped_midrules = self.gr_midrules()
        header_midrules = self.gr_midrules(add_first_col=False)

        return TableData(self.total_cols, 'c', self.top_header_name, groups, header, body, formatted_rank, pvalue, "2-%d"% self.total_cols, grouped_midrules, header_midrules)
        # return self.total_cols, 'c', self.top_header_name, groups, header, body, formatted_rank, pvalue, "2-%d"% self.total_cols, grouped_midrules, header_midrules
        #
        #
        # return '\\toprule \n' \
        #        '\\multicolumn{{{count}}}{{{alignment}}}{{{name}}} \\\\ \n' \
        #        '\\midrule \n' \
        #        '{groups} \\\\ \n' \
        #        '{header_midrule} \n' \
        #        '{header} \\\\ \n' \
        #        '{grouped_midrule} \n' \
        #        '{body} \\\\ \n' \
        #        '{grouped_midrule} \n' \
        #        '{rank} \\\\ \n' \
        #        '{grouped_midrule} \n' \
        #        '{pvalue} \\\\ \n' \
        #        '\\bottomrule \n'.format(count=self.total_cols, alignment='c', name=self.top_header_name, groups=groups,
        #                                 header=header, body=body, rank=formatted_rank, pvalue=pvalue, groups_range=
        #                                 "2-%d" % self.total_cols, grouped_midrule=grouped_midrules,
        #                                 header_midrule=header_midrules)


class MultiTable:
    def __init__(self):
        self.tables: [TableData] = list()

    def add_table(self, table: DataPivotTable):
        _table = table.build()
        if len(self.tables) > 0:
            _table = TableData(_table.cols, _table.alignment, _table.header_name, _table.groups, _table.header,
                               ['&' + row.split('&', 1)[1] for row in _table.body],
                               '&' + _table.formatted_rank.split('&', 1)[1],
                               '&' + _table.pvalue.split('&', 1)[1],
                               _table.groups_range, _table.grouped_midrule, _table.header_midrule)
        self.tables.append(_table)

    def compact(self, func: callable, sep: str='&'):
        return reduce(lambda x, y: x + sep + y, list(map(func, self.tables)))

    def compact2(self, func: callable, sep: str='&'):
        res = reduce(lambda x, y: x + sep + y, list(map(func, self.tables)))
        return reduce(lambda x, y: x + '\\\\\n' + y, res)

    def build(self) -> str:
        body = self.compact2(lambda x: x.body)
        rank = self.compact(lambda x: x.formatted_rank)
        pvalue = self.compact(lambda x: x.pvalue)


        top = """
        \\begin{tabular}{cccccccccccccccccccccc}
\cline{2-3} \cline{5-8} \cline{10-12} \cline{14-18} \cline{20-22} 
 & \multicolumn{2}{c}{(a) Standardization} &  & \multicolumn{4}{c}{(b) $n_{c}$ } &  & \multicolumn{3}{c}{(c) $(k_{min},k_{max})$} &  & \multicolumn{5}{c}{(d) $\sigma_{0}$} &  & \multicolumn{3}{c}{(e) $m$}\\tabularnewline
Problem & Off & On &  & $2n$ & $2n^{2}$ & $2^{n}$ & $n^{3}$ &  & $(1,1)$ & $(1,\infty)$ & $(2,\infty)$ &  & $0.125$ & $0.25$ & $0.5$ & $1.0$ & $2.0$ &  & $0.9$ & $1.0$ & $1.1$\\tabularnewline
\cline{1-3} \cline{5-8} \cline{10-12} \cline{14-18} \cline{20-22}
        """
        body = f'{body} \\\\'
        bottom = "\cline{1-3} \cline{5-8} \cline{10-12} \cline{14-18} \cline{20-22}\n" + \
                 f'{rank}\\tabularnewline\n' +\
                 '\cline{1-3} \cline{5-8} \cline{10-12} \cline{14-18} \cline{20-22}\n' +\
                 f'{pvalue}\\tabularnewline\n' +\
                 '\cline{1-3} \cline{5-8} \cline{10-12} \cline{14-18} \cline{20-22}\n' +\
                 '\end{tabular}\n'

        return top + body + bottom

    def print(self):
        print(self.build())


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
        return math('problem') + sep + reduce(lambda x, y: x + sep + y, attributes)

    @property
    def body(self):
        formatter = lambda x: "%0.3f %s" % (x[0], Formatter.format_error(x[1]))
        map_series = lambda s: Formatter.format_model_name(s.name) + sep + reduce(lambda xi, yi: xi + sep + yi, s) + row_end
        reducer = lambda r1, r2: r1 + eol + r2

        body = self.cm.applymap(formatter)
        body = body.apply(map_series, axis=1)
        body = reduce(reducer, body)

        return body

    def build(self) -> str:
        map_series = lambda s: str(s.name) + sep + reduce(lambda xi, yi: xi + sep + yi, s) + row_end
        footer = pd.Series(data=self.cm.applymap(lambda x: x[0]).mean().apply(lambda x: "%0.3f" % x), name='średnia')
        footer = map_series(footer)
        return '\\toprule\n' \
               '{header} \\\\\n' \
               '\\midrule\n' \
               '{body} \n' \
               '\\midrule\n' \
               '{footer} \n' \
               '\\bottomrule\n'.format(header=self.header, body=self.body, footer=footer)


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


class Attribute(Component):
    _template = '{}\n'


class Environment:
    def __init__(self):
        self._body: [str, type, dict] = ''
        self.component_type: [str, None] = None
        self.comment: CommentBlock = None
        self.bracket_options: Brackets = Brackets()
        self.curly_options: Curly = Curly()
        self.attribute: Attribute = Attribute()

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
        attribute = self.attribute.build()
        return '{comment}\n' \
               '{attribute}' \
               '\\begin{{{component}}}{curly}{brackets}\n' \
               '{body}\n' \
               '\\end{{{component}}}\n'.format(component=self.component_type, curly=curly_options,
                                               brackets=bracket_options, body=body, comment=comment,
                                               attribute=attribute)


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
        if isinstance(value, DataPivotTable):
            cols = ['r' * value.cols for _ in range(len(value.measures))]
            columns = 'l' + reduce(lambda x, y: x + "!{\color{white}\\vrule width 10pt}" + y, cols)
            self.curly_options.value = columns
            Environment.body.fset(self, value.build())
        elif isinstance(value, DataTable) or isinstance(value, ConfusionMatrix):
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
        # if isinstance(self.body, dict):
        #     items: dict = self.body
        #     mapped_items = list(
        #         map(lambda item: Comment(value="%25s\t\t%s" % (item[0], item[1])).build(), items.items()))
        #     return reduce(lambda x, y: x + y, mapped_items)
        return ''


class Experiment:
    def __init__(self, index: int, attribute, header, table, split, benchmark_mode=False, measure:[Measure]=[MeasureF],
                 reverse_colors: bool = False):
        self.index = index
        self.attribute = attribute
        self.header = header
        self.table = table
        self.split = split
        self.benchmark_mode = benchmark_mode
        self.measure = measure
        self.reverse_colors = reverse_colors

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
    tabular.attribute = Attribute(value="\\setlength{\\tabcolsep}{2pt}")
    tabular.body = data_table
    tabular.comment = CommentBlock(aggregator.info)

    log.info("Writing: %s" % experiment)

    # if len(sys.argv) > 1:
    log.debug("Finished: %s" % experiment)
    write_file(filename=filename, data=tabular.build(), path="./resources/")


def get_table_data(experiment: Experiment):
    aggregator = Aggragator(experiment=experiment.index, benchmark_mode=experiment.benchmark_mode,
                            attribute=experiment.attribute, measures=experiment.measure)

    data_frame = aggregator.transform(split=experiment.split)

    data_table = experiment.table(data_frame, experiment.header, attribute=aggregator.attribute,
                                  attribute_values=aggregator.attribute_values, reverse_colors=experiment.reverse_colors)
    return data_table


if __name__ == '__main__':
    experiment1 = Experiment(index=1, attribute='standardized', header=math('s'), table=DataPivotTable,
                             split=None)
    experiment2 = Experiment(index=2, attribute='constraints_generator', header=math('n_c'),
                             table=DataPivotTable, split=None)
    experiment3 = Experiment(index=3, attribute='clustering', header=math('(k_{min}, k_{max})'),
                             table=DataPivotTable, split=None)
    experiment4 = Experiment(index=4, attribute='sigma', header=math('\sigma_0'),
                             table=DataPivotTable, split=None)
    experiment5 = Experiment(index=5, attribute='margin', header=math('m'),
                             table=DataPivotTable, split=None)
    experiment6 = Experiment(index=6, attribute='train_sample', header=math('|X|'),
                             table=DataPivotTable, split=None, measure=[MeasureF1])
    experiment7 = Experiment(index=6, attribute='train_sample', header=math('|X|'),
                             table=DataPivotTable, split=None, measure=[MeasureTime], reverse_colors=True)

    multi_table = MultiTable()
    # [multi_table.add_table(get_table_data(experiment)) for experiment in [experiment1,
    #                                                                       experiment2,
    #                                                                       experiment3,
    #                                                                       experiment4,
    #                                                                       experiment5]]

    [multi_table.add_table(get_table_data(experiment)) for experiment in [experiment6,
                                                                          experiment7]]
    multi_table.print()

    # for experiment in [experiment3]:
    # for experiment in [experiment1]:
    #     table(experiment=experiment)
        # pass


    # aggregator = Aggragator('best')
    # confusion_matrix = ConfusionMatrix(aggregator.confusion_matrix())
    # cm_tabular = Tabular()
    # cm_tabular.body = confusion_matrix
    # write_tex_table(filename="cm", data=cm_tabular.build(), path='./resources/')
