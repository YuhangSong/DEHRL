import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class Visdomer(object):
    """docstring for Visdomer."""
    def __init__(self):
        super(Visdomer, self).__init__()

        from visdom import Visdom
        self.viz = Visdom(port=6006)
        self.win_dic = {}
        self.recorder = {
            'plot':{},
            'image':{},
            'text':{},
        }

    def clear_visdom_recorder(self):
        self.recorder = {
            'plot':{},
            'image':{},
            'text':{},
        }

    def record(self,name,value,data_type='plot'):
        if data_type=='plot':
            try:
                # try expend
                self.recorder[data_type][name] += [value]
            except Exception as e:
                # else, initialize
                self.recorder[data_type][name] = [value]
        else:
            self.recorder[data_type][name] = value

    def log_visdom(self):
        '''push everything to the visdom server'''

        # plot lines
        for plot_name in self.recorder['plot'].keys():
            if plot_name in self.win_dic.keys():
                if len(self.recorder['plot'][plot_name]) > 0:
                    self.win_dic[plot_name] = self.viz.line(
                        torch.from_numpy(np.asarray(self.recorder['plot'][plot_name])),
                        win=self.win_dic[plot_name],
                        opts=dict(title=TMUX+'\n'+plot_name)
                    )
            else:
                self.win_dic[plot_name] = None

        # log images
        for images_name in self.recorder['image'].keys():
            if images_name in self.win_dic.keys():
                self.win_dic[images_name] = self.viz.images(
                    self.recorder['image'][images_name],
                    win=self.win_dic[images_name],
                    opts=dict(title=images_name)
                )
            else:
                self.win_dic[images_name] = None

        # log text
        for text_name in self.recorder['text'].keys():
            if text_name in self.win_dic.keys():
                self.win_dic[text_name] = self.viz.text(
                    self.recorder['text'][text_name],
                    win=self.win_dic[text_name],
                    opts=dict(title=TMUX+'\n'+text_name)
                )
            else:
                self.win_dic[text_name] = None


def onehot_to_index(x):
    return np.where(x==1.0)[0][0]

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
