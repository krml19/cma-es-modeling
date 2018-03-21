from enum import Enum
import operator


class Operator(Enum):
    le = operator.le
    lt = operator.lt
    ge = operator.ge
    gt = operator.gt
    eq = operator.eq

    def describe(self):
        # self is the member here
        return self.name, self.value

    def __str__(self):
        return 'selected operator: {0}'.format(self.value)

    def compare(self, lhs: float, rhs: float) -> bool:
        return self.value(lhs, rhs)


class Constraint:
    weight = None

    def __init__(self, _operator: Operator, value: float, weight=1):
        self.weight = weight
        self._operator = _operator
        self.value = value

    def match(self, value) -> bool:
        return self._operator.compare(self.value, value)


class Constraints:
    def __init__(self, constraints):
        self.constraints = constraints

    def validate(self, value):
        return not (False in list(map(lambda constraint: constraint.match(value), self.constraints)))
