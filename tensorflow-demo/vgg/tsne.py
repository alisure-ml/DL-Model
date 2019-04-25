import multiprocessing
import numpy as np
import os
import alisuretool as alitool
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embedding(data):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    color = "bgrcmyk"
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], s="*", color=color[0], fontdict={'weight': 'bold', 'size': 6})
        pass

    plt.xticks([])
    plt.yticks([])
    return fig


def main(data_filename, result_png):
    print('Computing t-SNE embedding {} {}'.format(data_filename, result_png))

    data = np.load(data_filename)
    data = np.resize(data, new_shape=[len(data), data[0].size])

    print("cal tsne")
    tsne = TSNE(n_components=2)
    result = tsne.fit_transform(data)

    print("begin to embedding {}".format(result_png))
    fig = plot_embedding(result)

    print("begin to save {}".format(result_png))
    plt.savefig(result_png)
    plt.show(fig)
    pass


if __name__ == '__main__':
    # filename = "20181009_20181009090556_20181009090638_090323_224_224"
    # filename = "20181009_20181009090556_20181009090638_090323_448_448"
    filename = "20181009_20181009090613_20181009090627_090339_224_224"
    main(data_filename="../pool5/{}.pkl".format(filename), result_png="../pool5/{}.png".format(filename))
    pass
