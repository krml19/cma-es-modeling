import sqlite3
import numpy
import math
import scipy

__author__ = 'Tomasz Pawlak'


#
# SQLite missing functions are defined here
#

class Median:
    """Implements median function for SQLite"""

    def __init__(self):
        self.values = []

    def step(self, value):
        try:
            if value is not None:
                self.values.append(value)
        except Exception as e:
            print("Median.step: " + e)
            raise e

    def finalize(self):
        try:
            if len(self.values) == 0:
                return None
            m = numpy.median(self.values)
            self.values = []
            return m
        except Exception as e:
            print("Median.finalize: " + e)
            raise e

    def register(self, db):
        db.create_aggregate("median", 1, Median)


class MedianLowCF:
    """Implements low end of 95% confidence interval of median for SQLite"""

    def __init__(self):
        self.values = []

    def step(self, value):
        try:
            if value is not None:
                self.values.append(value)
        except Exception as e:
            print("MedianLowCF.step: " + e)
            raise e

    def finalize(self):
        try:
            if len(self.values) == 0:
                return None

            self.values = sorted(self.values)
            c = len(self.values)
            c = int(max(math.floor((float(c) - 1.959963985 * math.sqrt(c)) * 0.5), 0))
            m = self.values[c]
            self.values = []
            return m
        except Exception as e:
            print("MedianLowCF.finalize: " + e)
            raise e

    def register(self, db):
        db.create_aggregate("medianLowCF", 1, MedianLowCF)


class MedianHighCF:
    """Implements high end of 95% confidence interval of median for SQLite"""

    def __init__(self):
        self.values = []

    def step(self, value):
        try:
            if value is not None:
                self.values.append(value)
        except Exception as e:
            print("MedianHighCF.step: " + e)
            raise e

    def finalize(self):
        try:
            if len(self.values) == 0:
                return None

            self.values = sorted(self.values)
            c = len(self.values)
            c = int(min(math.ceil(1.0 + (float(c) + 1.959963985 * math.sqrt(c)) * 0.5), c - 1))
            m = self.values[c]
            self.values = []
            return m
        except Exception as e:
            print("MedianHighCF.finalize: " + e)
            raise e

    def register(self, db):
        db.create_aggregate("medianHighCF", 1, MedianHighCF)


class WeightedMedian(Median):
    """Implements weighted median function for SQLite"""

    def step(self, value, count):
        try:
            if value is not None:
                self.values += [value] * count
        except Exception as e:
            print("Median.step: " + e)
            raise e

    def register(self, db):
        db.create_aggregate("weightedMedian", 2, WeightedMedian)


class WeightedMedianLowCF(MedianLowCF):
    """Implements low end of 95% confidence interval of weighted median for SQLite"""

    def step(self, value, count):
        try:
            if value is not None:
                self.values += [value] * count
        except Exception as e:
            print("MedianLowCF.step: " + e)
            raise e

    def register(self, db):
        db.create_aggregate("weightedMedianLowCF", 2, WeightedMedianLowCF)


class WeightedMedianHighCF(MedianHighCF):
    """Implements high end of 95% confidence interval of weighted median for SQLite"""

    def step(self, value, count):
        try:
            if value is not None:
                self.values += [value] * count
        except Exception as e:
            print("MedianHighCF.step: " + e)
            raise e

    def register(self, db):
        db.create_aggregate("weightedMedianHighCF", 2, WeightedMedianHighCF)


class Sqrt:
    """Implements square root for SQLite"""

    @staticmethod
    def calculate(value):
        if value is None or value < -1e-12:
            return None
        elif value < 0.0:  # range -1e-12 .. 0.0 to handle numerical errors
            return 0.0
        return math.sqrt(value)

    def register(self, db):
        db.create_function("sqrt", 1, self.calculate)


class NormalDistTest:
    """Implements test for normal distribution for SQLite"""

    def __init__(self):
        self.values = []

    def step(self, value):
        try:
            if value is not None:
                self.values.append(value)
        except Exception as e:
            print("NormalDistTest.step: " + e)
            raise e

    def finalize(self):
        try:
            if len(self.values) == 0:
                return None

            # mean = numpy.mean(self.values)
            # stddev = numpy.std(self.values)
            # self.values = (self.values - mean)/stddev

            # self.values = scipy.stats.mstats.trim(self.values, limits=(0.025, 0.975), relative=True)

            # print len(self.values)
            # low = numpy.percentile(self.values, 0.005)
            # high = numpy.percentile(self.values, 0.995)
            # self.values = [x for x in self.values if low <= x and x <= high]
            # self.values = sorted(self.values)
            # self.values = self.values[int(len(self.values) * 0.025):int(len(self.values)*0.975)]
            self.values = scipy.stats.mstats.winsorize(self.values, limits=0.05)
            # self.values = scipy.stats.mstats.trim(self.values, limits=(0.025, 0.975), relative=True)

            while len(self.values) < 8:
                self.values = self.values + self.values

            test = scipy.stats.normaltest(self.values)
            # test = scipy.stats.jarque_bera(self.values)
            # print test
            self.values = []
            return test[1]
        except Exception as e:
            print("NormalDistTest.finalize: " + e)
            raise e

    def register(self, db):
        db.create_aggregate("NormalDistTest", 1, NormalDistTest)


class TTest:
    """Implements two-sided t-test for mean equal to the given value for SQLite"""

    def __init__(self):
        self.values = []
        self.expected_mean = 0

    def step(self, value, expected_mean):
        try:
            self.expected_mean = expected_mean
            if value is not None:
                self.values.append(value)
        except Exception as e:
            print("TTest.step: " + e)
            raise e

    def finalize(self):
        try:
            if len(self.values) == 0:
                return None

            mean = numpy.mean(self.values)
            stddev = numpy.std(self.values)
            n = len(self.values)

            if stddev == 0.0 and n > 1:
                self.values = []
                return 1.0 if abs(mean - self.expected_mean) < 1.0E-6 else 0.0

            tstat = (mean - self.expected_mean) * math.sqrt(n - 1) / stddev

            pvalue = scipy.stats.t.cdf(tstat, len(self.values) - 1)
            if pvalue > 0.5:
                pvalue = 1.0 - pvalue
            pvalue *= 2  # two sided test
            # print "mean: %f stdev: %f n: %d tstat: %f pvalue: %f" % (mean, stddev, len(self.values), tstat, pvalue)
            self.values = []
            return pvalue
        except Exception as e:
            print("TTest.finalize: " + e)
            raise e

    def register(self, db):
        db.create_aggregate("TTest", 2, TTest)


def register_aggregates(db):
    Median().register(db)
    MedianLowCF().register(db)
    MedianHighCF().register(db)
    WeightedMedian().register(db)
    WeightedMedianLowCF().register(db)
    WeightedMedianHighCF().register(db)
    Sqrt().register(db)
    NormalDistTest().register(db)
    TTest().register(db)


# Database management

def prepare_connection(filename: str) -> sqlite3.Connection:
    db = sqlite3.connect(filename, 3600.0, 0, "IMMEDIATE")
    cursor = db.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=OFF")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute("PRAGMA journal_mode=TRUNCATE")
    cursor.execute("PRAGMA page_size=" + str(1 << 15))
    cursor.execute("PRAGMA threads=3")
    register_aggregates(db)
    return db
