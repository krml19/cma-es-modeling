import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def draw2d(df, selected=[0, 1], constraints=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    red = (1.0, 0.0, 0.0, 0.5)
    green = (0.0, 1.0, 0.0, 0.5)
    blue = (0.0, 0.0, 1.0, 0.5)

    if 'valid' in df:
        colors = [green if v == True else red for v in df['valid'].values]
        sizes = [1 if v == True else 0.5 for v in df['valid'].values]
    else:
        colors = blue
        sizes = 0.5

    x_1 = df.columns[selected[0]]
    x_2 = df.columns[selected[1]]
    plt.scatter(x=df[x_1].values, y=df[x_2].values, c=colors, s=sizes)

    ax.set_xlabel(x_1)
    ax.set_ylabel(x_2)

    draw_cube2d_bounds(ax)
    draw_constraints(ax, constraints=constraints)
    plt.show()


def draw3d(df, selected=[0, 1, 2]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for x, y, z, v in zip(df[df.columns[selected[0]]].values, df[df.columns[selected[1]]].values, df[df.columns[selected[2]]].values, df['valid'].values):
        s = 1 if v == True else 0.5
        c = 'g' if v == True else 'r'
        ax.scatter(x, y, z, c=c, s=s)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def verts(bounds):
    x1 = bounds[0]
    x2 = bounds[1]
    return [(x1[0], x2[0]), (x1[1], x2[0]), (x1[1], x2[1]), (x1[0], x2[1]), (0, 0)]


def verts_codes():
    return [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]


def cube_rect():
    bounds = [[1, 3.7], [2, 7.4]]
    # bounds = [[-3, 3], [-5, 5]]

    return verts_codes(), verts(bounds)


def draw_cube2d_bounds(ax):
    codes, vertices = cube_rect()

    vertices = np.array(vertices, float)
    path = Path(vertices, codes)

    pathpatch = PathPatch(path, facecolor='None', edgecolor='blue')

    ax.add_patch(pathpatch)
    ax.set_facecolor((1, 1, 1, 0.5))


def draw_constraints(ax, constraints):
    if constraints is None:
        return

    for constraint in constraints:
        normalized = np.linalg.norm(constraint)
        x0 = constraint[0] / normalized
        x1 = constraint[1] / normalized
        ax.plot([0, x0], [0, x1], 'k-')

