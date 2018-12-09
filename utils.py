import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class ObsNorm(object):
    """docstring for ObsNorm."""
    def __init__(self, env, save_dir, nsteps=10000):
        super(ObsNorm, self).__init__()
        self.env = env
        self.nsteps = nsteps
        self.save_dir = save_dir

        self.restore()

    def random_agent_ob_mean_std(self):
        print('Running random_agent_ob_mean_std for nsteps: {}'.format(self.nsteps))
        ob = np.asarray(self.env.reset())
        obs = [ob]
        for _ in range(self.nsteps):
            ac = self.env.action_space.sample()
            ob, _, done, _ = self.env.step(ac)
            if done:
                ob = self.env.reset()
            obs.append(np.asarray(ob))

        '''original bound'''

        self.mean = np.mean(obs, 0).astype(np.float32)
        '''after -mean, it should be between -255 ~ +255'''

        self.std = np.std(obs, 0).mean().astype(np.float32)
        '''after /std, it should be between -255/std, 255/std'''
        self.bound = np.array([-255.0/self.std,255.0/self.std])

    def store(self):
        try:
            np.save(
                self.save_dir+'/ob_mean.npy',
                self.mean,
            )
            np.save(
                self.save_dir+'/ob_std.npy',
                self.std,
            )
            np.save(
                self.save_dir+'/ob_bound.npy',
                self.bound,
            )
            print('{}: Store Successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('{}: Store Failed. Due to {}'.format(self.__class__.__name__,e))

    def restore(self):
        try:
            self.mean  = np.load(self.save_dir+'/ob_mean.npy')
            self.std   = np.load(self.save_dir+'/ob_std.npy')
            self.bound = np.load(self.save_dir+'/ob_bound.npy')
            print('{}: Restore Successed.'.format(self.__class__.__name__))
        except Exception as e:
            print('{}: Restore Failed. Due to {}'.format(self.__class__.__name__,e))
            self.random_agent_ob_mean_std()
            self.store()
        print('Estimated mean {} shape {}; std {} bound {}'.format(self.mean,self.mean.shape,self.std, self.bound))

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
