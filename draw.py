import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import axes3d, Axes3D
import pandas as pd

def draw2d(df, selected=[0, 1], constraints=None, title=None):
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(111)
    draw2dset(ax=ax, df=df, selected=selected, constraints=constraints, title=title)
    plt.show()


def draw2dset(df, ax=None, selected=[0, 1], constraints=None, title=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')
    red = (1.0, 0.0, 0.0, 0.1)
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
    # plt.show()


def draw3dset(df, ax=None, selected=[0, 1, 2], constraints=None, title=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')
    red = (1.0, 0.0, 0.0, 0.1)
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
    x_3 = df.columns[selected[2]]
    ax.scatter(xs=df[x_1].values, ys=df[x_2].values, zs=df[x_3].values, c=colors, s=sizes)

    ax.set_xlabel(x_1)
    ax.set_ylabel(x_2)
    ax.set_zlabel(x_3)
    if title is not None:
        ax.set_title(title)


def draw2dmodel(train, df, model, constraints=None, title=None):
    fig = plt.figure(figsize=(8, 12))
    gs1 = gridspec.GridSpec(nrows=3, ncols=2)
    valid_ax = fig.add_subplot(gs1[:-1, :])
    train_ax = fig.add_subplot(gs1[-1, :-1])
    w_ax = fig.add_subplot(gs1[-1, -1])

    # validation set
    draw2dset(valid_ax, df=df, title=title)

    # train set
    train_title = 'Train set ({}/{})'.format(train['valid'].sum(),
                                             train['valid'].count()) if 'valid' in train else 'Train set'
    draw2dset(train_ax, df=train, title=train_title)
    if model == 'cube':
        draw_cube(ax=train_ax)
    elif model == 'ball':
        draw_ellipse(train_ax, width=5.4, height=5.4)
    elif model == 'simplex':
        draw_simplex(train_ax)
    train_ax.autoscale()

    # constraints
    draw_constraints(w_ax, constraints=constraints, title='w (n={})'.format(len(constraints)))

    plt.show()


def draw3dmodel(train, df, model, constraints=None, title=None):
    fig = plt.figure(figsize=(8, 12))
    gs1 = gridspec.GridSpec(nrows=3, ncols=2)
    valid_ax = fig.add_subplot(gs1[:-1, :], projection='3d')
    train_ax = fig.add_subplot(gs1[-1, :-1], projection='3d')
    w_ax = fig.add_subplot(gs1[-1, -1], projection='3d')

    # validation set
    draw3dset(valid_ax, df=df, title=title)

    # train set
    train_title = 'Train set ({}/{})'.format(train['valid'].sum(), train['valid'].count()) if 'valid' in train else 'Train set'

    draw3dset(train_ax, df=train, title=train_title)

    # constraints
    draw_constraints(w_ax, constraints=constraints, title='w (n={})'.format(len(constraints)))
    plt.show()


def draw3d(df, selected=[0, 1, 2], title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    draw3dset(ax=ax, df=df, title='Model')
    x_0 = df.columns[selected[0]]
    x_1 = df.columns[selected[1]]
    x_2 = df.columns[selected[2]]

    ax.set_xlabel(x_0)
    ax.set_ylabel(x_1)
    ax.set_zlabel(x_2)

    if title is not None:
        ax.set_title(title)

    plt.show()


def draw_cube(ax):

    bounds = [[1, 3.7], [2, 7.4]]
    x1 = bounds[0]
    x2 = bounds[1]
    points = [(x1[0], x2[0]), (x1[1], x2[0]), (x1[1], x2[1]), (x1[0], x2[1])]
    rectangle = Polygon(points, closed=True, fill=None, edgecolor='blue')
    ax.add_patch(rectangle)


def draw_constraints(ax, constraints, title=None):
    draw_ellipse3d(ax)
    if constraints is None:
        return

    for constraint in constraints:
        normalized = np.linalg.norm(constraint)
        x0 = constraint[0] / normalized
        x1 = constraint[1] / normalized
        if len(constraint) == 3:
            x2 = constraint[2] / normalized
            ax.plot([0, x0], [0, x1], [0, x2], 'k-')
            ax.scatter(xs=x0, ys=x1, zs=x2, color=(1.0, 0.0, 0.0, 0.7))
        else:
            ax.scatter(x=x0, y=x1, color=(1.0, 0.0, 0.0, 0.7))
            ax.plot([0, x0], [0, x1], 'k-')

        ax.set_xlabel('x_0')
        ax.set_ylabel('x_1')
        if hasattr(ax, 'set_zlabel'):
            ax.set_zlabel('x_2')
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


def draw_ellipse3d(ax):
    # Make data
    count = 60
    u = np.linspace(0, 2 * np.pi, count)
    v = np.linspace(0, np.pi, count)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    if hasattr(ax, 'plot_wireframe'):
        ax.plot_wireframe(x, y, z, color=(0.0, 0.0, 1.0, 0.1))
    else:
        ellipse = Ellipse(xy=(0, 0), width=2, height=2, fill=False, edgecolor=(0.0, 0.0, 1.0, 0.1))
        ax.add_patch(ellipse)


def draw_df(filename: str):
    df = pd.read_csv(filename)

    n = df.shape[1]
    if n > 3 or n < 2:
        print("Cannot draw.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') if n == 3 else fig.add_subplot(111)

    red = (1.0, 0.0, 0.0, 0.1)
    green = (0.0, 1.0, 0.0, 0.5)
    blue = (0.0, 0.0, 1.0, 0.5)

    colors = green
    sizes = 1

    if n == 2:
        draw2d_df(df, ax, colors, sizes)
    elif n == 3:
        draw3d_df(df, ax, colors, sizes)
    title = filename.split('/')[-1].split('_')
    title = "{name} [k={k}, n={n}]".format(name=title[1], k=title[2], n=title[3])
    ax.set_title(title)

    plt.show()


def draw2d_df(df, ax, colors, sizes):
    x_1 = df.columns[0]
    x_2 = df.columns[1]
    ax.scatter(x=df[x_1].values, y=df[x_2].values, c=colors, s=sizes)

    ax.set_xlabel(x_1)
    ax.set_ylabel(x_2)


def draw3d_df(df, ax, colors, sizes):
    x_1 = df.columns[0]
    x_2 = df.columns[1]
    x_3 = df.columns[2]
    ax.scatter(xs=df[x_1].values, ys=df[x_2].values, zs=df[x_3].values, c=colors, s=sizes)

    ax.set_xlabel(x_1)
    ax.set_ylabel(x_2)
    ax.set_zlabel(x_3)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# draw_ellipse3d(ax)
# plt.ion()
# plt.show()