import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon


def draw2d(df, selected=[0, 1], constraints=None, title=None):
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(111)
    draw2dset(ax=ax, df=df, selected=selected, constraints=constraints, title=title)


def draw2dset(ax, df, selected=[0, 1], constraints=None, title=None):
    # ax.set_aspect('equal', 'box')
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
    ax.scatter(x=df[x_1].values, y=df[x_2].values, c=colors, s=sizes)

    ax.set_xlabel(x_1)
    ax.set_ylabel(x_2)
    if title is not None:
        ax.set_title(title)


def draw2dmodel(train, df, model, constraints=None, title=None):
    fig = plt.figure(figsize=(8, 12))
    gs1 = gridspec.GridSpec(nrows=3, ncols=2)
    ax1 = fig.add_subplot(gs1[:-1, :])
    ax2 = fig.add_subplot(gs1[-1, :-1])
    ax3 = fig.add_subplot(gs1[-1, -1])

    # validation set
    draw2dset(ax1, df=df, title=title)

    # train set
    draw2dset(ax2, df=train, title='Train set')
    if model == 'cube':
        draw_cube(ax=ax2)
    elif model == 'ball':
        draw_ellipse(ax2, width=5.4, height=5.4)
    elif model == 'simplex':
        draw_simplex(ax2)
    ax2.autoscale()

    # constraints
    draw_constraints(ax3, constraints=constraints, title='w')

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


def draw_cube(ax):

    bounds = [[1, 3.7], [2, 7.4]]
    x1 = bounds[0]
    x2 = bounds[1]
    points = [(x1[0], x2[0]), (x1[1], x2[0]), (x1[1], x2[1]), (x1[0], x2[1])]
    rectangle = Polygon(points, closed=True, fill=None, edgecolor='blue')
    ax.add_patch(rectangle)


def draw_constraints(ax, constraints, title=None):
    if constraints is None:
        return

    for constraint in constraints:
        normalized = np.linalg.norm(constraint)
        x0 = constraint[0] / normalized
        x1 = constraint[1] / normalized
        ax.plot([0, x0], [0, x1], 'k-')
        if title is not None:
            ax.set_title(title)


def draw_ellipse(ax, xy=(1, 2), width: float=1, height: float=1):
    ellipse = Ellipse(xy=xy, width=width, height=height, fill=False, edgecolor='blue')
    ax.add_patch(ellipse)


def draw_simplex(ax):
    x1 = 2.55
    x2 = 0.18
    points = [[0, 0], [x1, x2], [x2, x1]]
    triangle = Polygon(points, closed=True, fill=None, edgecolor='blue')
    ax.add_patch(triangle)
