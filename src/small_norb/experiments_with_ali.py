import torch
import torch.nn as nn
import numpy as np

from torch import optim

from ..sharedCode.provider import Provider
from ..sharedCode.experiments import train_test_from_dataset, \
    UpperDiagonalThresholdedLogTransform, \
    pers_dgm_center_init,\
    SLayerPHT

import chofer_torchex.utils.trainer as tr
from chofer_torchex.utils.trainer.plugins import *


import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
from itertools import *


def _parameters():
    return {
        'data_path': None,
        'epochs': 300,
        'momentum': 0.7,
        'lr_start': 0.1,
        'lr_ep_step': 20,
        'lr_adaption': 0.5,
        'test_ratio': 0.5,
        'batch_size': 128,
        'cuda': False
    }

def _data_setup(params):
    view_name_template = 'dim_0_dir_{}'
    subscripted_views = sorted([view_name_template.format(i) for i in range(32)])
    assert (str(len(subscripted_views)) in params['data_path'])

    print('Loading provider...')
    dataset = Provider()
    dataset.read_from_h5(params['data_path'])

    assert all(view_name in dataset.view_names for view_name in subscripted_views)

    print('Create data loader...')
    data_train, data_test = train_test_from_dataset(dataset,
                                                    test_size=params['test_ratio'],
                                                    batch_size=params['batch_size'])

    return data_train, data_test, subscripted_views



class MyArchitecture(object):
    def __init__(self):
        mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
        mb_size = 32
        z_dim = 10
        X_dim = mnist.train.images.shape[1]
        y_dim = mnist.train.labels.shape[1]
        h_dim = 128
        cnt = 0
        lr = 1e-3


        # Inference net (Encoder) Q(z|X)
        Q = torch.nn.Sequential(
            torch.nn.Linear(X_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, z_dim)
        )

        # Generator net (Decoder) P(X|z)
        P = torch.nn.Sequential(
            torch.nn.Linear(z_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, X_dim),
            torch.nn.Sigmoid()
        )

        D_ = torch.nn.Sequential(
            torch.nn.Linear(X_dim + z_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 1),
            torch.nn.Sigmoid()
        )

    def log(self, x):
        return torch.log(x + 1e-8)

    def D(self, X, z):
        return D_(torch.cat([X, z], 1))

    def reset_grad(self):
        Q.zero_grad()
        P.zero_grad()
        D_.zero_grad()

    def compile(self):

        G_solver = optim.Adam(chain(Q.parameters(), P.parameters()), lr=lr)
        D_solver = optim.Adam(D_.parameters(), lr=lr)

        for it in range(50):
            # Sample data
            z = Variable(torch.randn(mb_size, z_dim))
            X, _ = mnist.train.next_batch(mb_size)
            X = Variable(torch.from_numpy(X))

            # Discriminator
            z_hat = Q(X)
            X_hat = P(z)

            D_enc = D(X, z_hat)
            D_gen = D(X_hat, z)

            D_loss = -torch.mean(log(D_enc) + log(1 - D_gen))  #minimax function

            D_loss.backward()
            D_solver.step()
            G_solver.step()
            reset_grad()

            # Autoencoder Q, P
            z_hat = Q(X)
            X_hat = P(z)

            D_enc = D(X, z_hat)
            D_gen = D(X_hat, z)

            G_loss = -torch.mean(log(D_gen) + log(1 - D_enc))

            G_loss.backward()
            G_solver.step()
            reset_grad()

            # Print and plot every now and then
            if it % 1000 == 0:
                print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
                      .format(it, D_loss.data[0], G_loss.data[0]))

                samples = P(z).data.numpy()[:16]

                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(4, 4)
                gs.update(wspace=0.05, hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')

                plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
                cnt += 1
                plt.close(fig)

class MyModel(torch.nn.Module):
    def __init__(self, subscripted_views):
        super(MyModel, self).__init__()
        self.subscripted_views = subscripted_views

        n_elements = 75
        n_filters = 32
        stage_2_out = 25
        n_neighbor_directions = 1

        self.transform = UpperDiagonalThresholdedLogTransform(0.1)

        self.pht_sl = SLayerPHT(len(subscripted_views),
                                n_elements,
                                2,
                                n_neighbor_directions=n_neighbor_directions,
                                center_init=self.transform(pers_dgm_center_init(n_elements)),
                                sharpness_init=torch.ones(n_elements, 2) * 4)

        self.stage_1 = []
        for i in range(len(subscripted_views)):
            seq = torch.nn.Sequential()
            seq.add_module('conv_1', torch.nn.Conv1d(1 + 2 * n_neighbor_directions, n_filters, 1, bias=False))
            seq.add_module('conv_2', torch.nn.Conv1d(n_filters, 8, 1, bias=False))
            self.stage_1.append(seq)
            self.add_module('stage_1_{}'.format(i), seq)

        self.stage_2 = []
        for i in range(len(subscripted_views)):
            seq = torch.nn.Sequential()
            seq.add_module('linear_1', torch.nn.Linear(n_elements, stage_2_out))
            seq.add_module('batch_norm', torch.nn.BatchNorm1d(stage_2_out))
            seq.add_module('linear_2'
                           , torch.nn.Linear(stage_2_out, stage_2_out))
            seq.add_module('relu', torch.nn.ReLU())
            seq.add_module('Dropout', torch.nn.Dropout(0.4))

            self.stage_2.append(seq)
            self.add_module('stage_2_{}'.format(i), seq)

        linear_1 = torch.nn.Sequential()
        linear_1.add_module('linear', torch.nn.Linear(len(subscripted_views) * stage_2_out, 50))
        linear_1.add_module('batchnorm', torch.nn.BatchNorm1d(50))
        linear_1.add_module('drop_out', torch.nn.Dropout(0.3))
        self.linear_1 = linear_1

        linear_2 = torch.nn.Sequential()
        linear_2.add_module('linear', torch.nn.Linear(50, 20))

        self.linear_2 = linear_2

    def forward(self, batch):
        x = [batch[n] for n in self.subscripted_views]
        x = [[self.transform(dgm) for dgm in view_batch] for view_batch in x]

        x = self.pht_sl(x)

        x = [l(xx) for l, xx in zip(self.stage_1, x)]

        x = [torch.squeeze(torch.max(xx, 1)[0]) for xx in x]

        x = [l(xx) for l, xx in zip(self.stage_2, x)]

        x = torch.cat(x, 1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x

def _create_trainer(model, params, data_train, data_test):
    optimizer = optim.SGD(model.parameters(),
                      lr=params['lr_start'],
                      momentum=params['momentum'])

    loss = torch.
    nn.CrossEntropyLoss()

    trainer = tr.Trainer(model=model,
                     optimizer=optimizer,
                     loss=loss,
                     train_data=data_train,
                     n_epochs=params['epochs'],
                     cuda=params['cuda'],
                     variable_created_by_model=True)

    def determine_lr(self, **kwargs):
        epoch = kwargs['epoch_count']
        if epoch % params['lr_ep_step'] == 0:
            return params['lr_start'] / 2 ** (epoch / params['lr_ep_step'])

    lr_scheduler = LearningRateScheduler(determine_lr, verbose=True)
    lr_scheduler.register(trainer)

    progress = ConsoleBatchProgress()
    progress.register(trainer)

    prediction_monitor_test = PredictionMonitor(data_test,
                                                verbose=True,
                                                eval_every_n_epochs=1,
                                                variable_created_by_model=True)
    prediction_monitor_test.register(trainer)
    trainer.prediction_monitor = prediction_monitor_test

    return trainer


def experiment(data_path):
    params = _parameters()
    params['data_path'] = data_path

    if torch.cuda.is_available():
        params['cuda'] = True

    print('Data setup...')
    data_train, data_test, subscripted_views = _data_setup(params)

    print('Create model...')
    model = MyModel(subscripted_views)

    print('Setup trainer...')
    trainer = _create_trainer(model, params, data_train, data_test)
    print('Starting...')
    trainer.run()

    last_10_accuracies = list(trainer.prediction_monitor.accuracies.values())[-10:]
    mean = np.mean(last_10_accuracies)

    return mean
