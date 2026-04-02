import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def main():
    fig = plt.figure(figsize=plt.figaspect(0.5))

    points_3d_path = 'points_3d.json'

    with open(points_3d_path, 'r') as f:
        points_3d = np.array(json.load(f))


    x = points_3d[:,0].flatten()
    y = points_3d[:,1].flatten()
    z = points_3d[:,2].flatten()

    tri = mtri.Triangulation(x, y)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface Plot of 3D Points')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(-x, -y, -z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Scatter Plot of 3D Points')

    plt.show()

if __name__ == '__main__':
    main()