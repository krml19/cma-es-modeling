from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def draw2d(df, selected=[0, 1]):
    cols = df.columns
    color = np.where(df['valid'].values == True, 'g', 'r')
    df.plot(kind='scatter', x=cols[selected[0]], y=cols[selected[1]], c=color)
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