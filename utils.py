import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def list_to_str(x):
    return str(x).replace('[','').replace(']','').replace(', ','_')

def figure_to_array(fig):
    canvas=fig.canvas
    buf = io.BytesIO()
    canvas.print_png(buf)
    data=buf.getvalue()
    buf.close()
    buf=io.BytesIO()
    buf.write(data)
    img=Image.open(buf)
    img = np.asarray(img)
    return img

def actions_onehot_visualize(actions_onehot,figsize):
    actions_onehot = np.flip(actions_onehot, 0)
    plt.clf()
    fig, ax = plt.subplots(frameon=False,figsize=(figsize[0]/100.0, figsize[1]/100.0))
    im = ax.imshow(actions_onehot)
    ax.set_axis_off()
    img = figure_to_array(plt.gcf())[:,:,:-1]
    plt.close()
    return img

def action_to_onehot(action, action_space):
    onehot = np.zeros((action_space.n))
    onehot[action] = 1.0
    return onehot

def gray_to_rgb(image):
    image = np.expand_dims(image,2)
    return np.concatenate((image,image,image),2)

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
