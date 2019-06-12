from collections import OrderedDict
from sqlite3 import Connection
import re

from db import prepare_connection

objective = '''Maximize

\ Income R2=0.788
\ 93722.162654155 A + 21165.877947956 V -27094.8125944111 B + 10.3234375641855 S + 730.150534432567 U + 779.050350503559 Ph + 8.53640456601618 Pe + 68.0434445121963 L + [ -21605.5391379848 A ^ 2 -0.624614849392192 U ^ 2 ]
\ Cost R2=0.978
\ -149.509137264007 S + 49.7625466504911 U + 257.981212472661 Ph + 92.7577915132769 Pe + 77.9785094668861 L
\ combined
\ 93722.162654155 A + 21165.877947956 V -27094.8125944111 B + 159.8325748 S + 680.3879878 U + 521.069138 Ph -84.22138695 Pe -9.935064955 L + [ -21605.5391379848 A ^ 2 -0.624614849392192 U ^ 2 ]

\ Income R2=0.788, no seed
\ 93740.7806239932 A + 21204.5556123334 V -27075.1453001134 B + 731.11492471454 U + 779.351441082523 Ph + 8.53855870689193 Pe + 68.2243059839734 L + [ -21599.2659365671 A^2 -0.625429424827357 U^2 ]
\ Cost R2=0.978
\ -149.509137264007 S + 49.7625466504911 U + 257.981212472661 Ph + 92.7577915132769 Pe + 77.9785094668861 L
\ combined
93740.7806239932 A + 21204.5556123334 V -27075.1453001134 B + 149.509137264007 S + 681.3523781 U + 521.3702286 Ph -84.21923281 Pe -9.754203483 L + [ -21599.2659365671 A^2 -0.625429424827357 U^2 ]
'''

def process_cluster(db: Connection, num: int, parent: int, variables: dict):
    cur = db.execute("SELECT w_mathematica FROM cluster_%d WHERE parent=?" % num, [parent])
    constraints: str = cur.fetchone()[0]
    constraints = constraints.replace("&&", "")
    constraints = re.sub(r"[-]?\d+(\.\d+)?\s*$", (lambda val: str(float(val.group(0)) + 1e6)), constraints, flags=re.MULTILINE)

    constraints = constraints.replace("<", "+ 1e6 b%d  <" % num)
    for i, v in enumerate(variables):
        constraints = constraints.replace("x[%d]" % (i + 1), " %s" % v)
    return constraints


def produce_constraints(db: Connection, id: int, variables: dict):
    constraints = ""
    clusters = db.execute("SELECT clusters FROM experiments WHERE id=?", [id]).fetchone()[0]
    for c in range(clusters):
        constraints += process_cluster(db, c, id, variables) + "\n"
    constraints += " + ".join("b%d" % c for c in range(clusters)) + " >= 1\n"
    return constraints, clusters


def write_lp(filename: str, variables: dict, auxbinnum: int, constraints: str):
    with open("case_study.lp", "w") as fout:
        fout.write(objective)
        fout.write("Subject To\n")
        fout.write(constraints)
        fout.write("Bounds\n")
        for v, (domain, _min, _max) in variables.items():
            fout.write("%f <= %s <= %f\n" % (_min, v, _max))
        fout.write("Generals\n")
        for v, (domain, _min, _max) in variables.items():
            if domain == 'I':
                fout.write("%s\n" % v)
        fout.write("Binary\n")
        for aux in range(auxbinnum):
            fout.write("b%d\n" % aux)
        fout.write("End\n")

def main():
    id = 6565
    variables = OrderedDict(A=('R', 0.014, 5.322), V=('I', 0, 2), B=('I', 0, 2), S=('R', 4, 371.428571428571), U=('R', 0.874890638670166, 877.19298245614), Ph=('R', 0, 877.19298245614),
                            Pe=('R', 0, 43697.0338983051), L=('R', 108, 4551.72413793103))
    db = prepare_connection("experiments.sqlite")

    constraints, clusters = produce_constraints(db, id, variables)
    write_lp("case_study.lp", variables, clusters, constraints)


if __name__ == '__main__':
    main()
