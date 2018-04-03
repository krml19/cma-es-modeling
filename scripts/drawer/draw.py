import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def draw2d(df, selected=[0, 1]):
    cols = df.columns

    color = np.where(df['valid'].values == True, 'g', 'r') if 'valid' in df else 'b'
    df.plot(kind='scatter', x=cols[selected[0]], y=cols[selected[1]], c=color)

    draw_cube2d_bounds()
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


def verts(i, d=2.7):
    return [(i, i), (i, i+i*d), (i+i*d, i+i*d), (i+i*d, i), (0, 0)]


def verts_codes():
    return [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]


def draw_cube2d_bounds():
    codes = verts_codes()
    vertices = verts(1)

    vertices = np.array(vertices, float)
    path = Path(vertices, codes)

    pathpatch = PathPatch(path, facecolor='None', edgecolor='blue')

    fig, ax = plt.subplots()
    ax.add_patch(pathpatch)
    ax.set_title('A compound path')

    ax.dataLim.update_from_data_xy(vertices)
    ax.autoscale_view()