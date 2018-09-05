import numpy as np
from torch.nn import init
import matplotlib.pyplot as plt

def gaussian_intiailize(model, std=.01):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'fc' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters()
    ]

    for p in parameters:
        if p.dim() >= 2:
            init.normal(p, std=std)
        else:
            init.constant(p, 0)

def make_plots(imgs, path=None):
    imgs = imgs
    fig = plt.figure()
    for idx in range(16):
        ax = fig.add_subplot(4, 4, idx+1) # this line adds sub-axes
        ax.xaxis.set_visible(False) # same for y axis.
        ax.yaxis.set_visible(False) # same for y axis.
        ax.set_aspect('auto')
        img = imgs[idx]
        img = img.transpose(2, 1, 0)
        img = np.rot90(np.rot90(np.rot90(img)))

        ax.imshow(img) # this line creates the image using the pre-defined sub axes
    plt.savefig(path+".png", transparent=True)

    plt.show()
