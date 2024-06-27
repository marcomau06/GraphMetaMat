from src.generative_curve.test import test, plot_pred
# from src.utils import log_dir, writer
from src.utils import log_dir
from src.config import args
from pprint import pprint
from tqdm import tqdm

import torch.nn as nn
import os
import optuna
import copy

class TrainConfig:
    def __init__(self,
                 num_epochs, num_epochs_per_valid,
                 # best_checkpoint_metric,
                 best_checkpoint_metric_shape,
                 best_checkpoint_metric_magnitude,
                 no_epoch_checkpoints,
                 clip_grad, use_snapshot,
                 plot_num_samples_max, **kwargs):
        self.num_epochs = num_epochs
        # self.save_interval = save_interval
        self.num_epochs_per_valid = num_epochs_per_valid
        # self.best_checkpoint_metric = best_checkpoint_metric
        self.best_checkpoint_metric_magnitude = best_checkpoint_metric_magnitude
        self.best_checkpoint_metric_shape = best_checkpoint_metric_shape
        self.no_epoch_checkpoints = no_epoch_checkpoints
        self.use_snapshot = use_snapshot
        self.clip_grad = clip_grad
        self.plot_num_samples_max = plot_num_samples_max


def validate(dataset_valid, model, metric_best, train_cfg, device):
    print('running validation...')
    metrics_collated, metrics_raw, graph_li, curve_pred_li, curve_true_li, *_ = \
        test(dataset_valid, dataset_train, model, device, plot_num_samples_max=None)
    model.train()

    is_best_iter = metrics_collated[train_cfg.best_checkpoint_metric][0] <= metric_best
    if is_best_iter:
        metric_best = metrics_collated[train_cfg.best_checkpoint_metric][0]        
    return metric_best, is_best_iter

def train_hp_tuning(dataset_train, dataset_valid, model, optimizer, train_cfg, batch_size, trial, device):
    import time
    model.train()
    model = model.to(device)

    if train_cfg.use_snapshot is not None:
        num_snapshots = train_cfg.use_snapshot['num_snapshots']
        t_mult = train_cfg.use_snapshot['t_mult']
        restart_period_min = \
            int(
                train_cfg.num_epochs/
                (sum([int(t_mult**k) for k in range(num_snapshots)])+num_snapshots)
            )
        restart_period_max = \
            int(
                train_cfg.num_epochs/
                (sum([int(t_mult**k) for k in range(num_snapshots)]))
            )

        def get_num_epochs_all(restart_period, t_mult, num_snapshots):
            num_epochs_all = [restart_period]
            for k in range(num_snapshots-1):
                num_epochs_all.append(math.ceil(num_epochs_all[-1] * t_mult))
            num_epochs_all = sum(num_epochs_all)
            return num_epochs_all

        if restart_period_min == restart_period_max:
            restart_period = restart_period_max
            num_epochs_all = get_num_epochs_all(restart_period, t_mult, num_snapshots)
        else:
            assert restart_period_min < restart_period_max
            restart_period, num_epochs_all = None, None
            for restart_period in range(restart_period_max, restart_period_min, -1):
                num_epochs_all = get_num_epochs_all(restart_period, t_mult, num_snapshots)
                if num_epochs_all < train_cfg.num_epochs:
                    break
        print(f'restart_period={restart_period}epochs; {num_snapshots}-snapshots={num_epochs_all}epochs')
        scheduler = \
            CyclicLRWithRestarts(
                optimizer, batch_size, batch_size*len(dataset_train),
                restart_period=restart_period, t_mult=t_mult)
        restart_count = None
    else:
        scheduler, restart_count = None, None

    best_iter = -1
    num_iters = 0
    metric_best = +float('inf')
    loss_train = []
    save_model_li = []
    t0 = time.time()
    for num_epochs in range(train_cfg.num_epochs):
        save_snapshot = False
        if train_cfg.use_snapshot:
            assert scheduler is not None
            scheduler.step()
            if restart_count is not None:
                save_snapshot = restart_count != scheduler.restarts
            restart_count = scheduler.restarts

        for i, (graph, curve) in enumerate(dataset_train):
            
            if i * args['dataset']['batch_size'] >= 200000:
               break       
            
            model.train()
            optimizer.zero_grad()
            graph.to(device), curve.to(device)
            loss = model(graph, curve)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad)
            optimizer.step()
            if train_cfg.use_snapshot:
                scheduler.batch_step()

            # Store training loss values
            loss_train.append(float(loss.detach().cpu()))

            if num_iters % 100 == 0:
                frac = f' {i}/{len(dataset_train)}' if i != 0 else ''
                print(f'loss [{num_epochs}{frac} Epochs]: {float(loss.detach().cpu())}')

            if num_iters % (train_cfg.num_epochs_per_valid*len(dataset_train)) == 0:
                model.train()
                print(f'{train_cfg.num_epochs_per_valid} iterations took {time.time()-t0}s')
                metric_best, is_best_iter = \
                    validate(dataset_valid, model, metric_best, train_cfg, device)
                best_iter = num_iters if is_best_iter else best_iter
                t0 = time.time()

            # if num_iters % train_cfg.save_interval == 0:
            #     write_model(model, log_dir, num_iters)            
            
            num_iters += 1

        # if train_cfg.use_snapshot and save_snapshot:
        #     # print(f'snapshotting mode at epoch {epoch}')
        #     with torch.no_grad():
        #         model_snapshot = copy.deepcopy(model)
        #     save_model_li.append(model_snapshot)

        # if not train_cfg.no_epoch_checkpoints:
        #     write_model(model, log_dir, f'epoch_{num_epochs}')

        trial.report(metric_best, num_epochs)     
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return metric_best, loss_train


def read_model(model, dn, run_name):
    model.load_state_dict(torch.load(os.path.join(dn, f'model_{run_name}.pt')))


def write_model(model, dn, run_name):
    torch.save(model.state_dict(), os.path.join(dn, f'model_{run_name}.pt'))


#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################

# source: https://github.com/mpyrozhok/adamwr/blob/master/cyclic_scheduler.py
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import torch


class ReduceMaxLROnRestart:
    def __init__(self, ratio=0.75):
        self.ratio = ratio

    def __call__(self, eta_min, eta_max):
        return eta_min, eta_max * self.ratio


class ExpReduceMaxLROnIteration:
    def __init__(self, gamma=1):
        self.gamma = gamma

    def __call__(self, eta_min, eta_max, iterations):
        return eta_min, eta_max * self.gamma ** iterations


class CosinePolicy:
    def __call__(self, t_cur, restart_period):
        return 0.5 * (1. + math.cos(math.pi *
                                    (t_cur / restart_period)))


class ArccosinePolicy:
    def __call__(self, t_cur, restart_period):
        return (math.acos(max(-1, min(1, 2 * t_cur
                                      / restart_period - 1))) / math.pi)


class TriangularPolicy:
    def __init__(self, triangular_step=0.5):
        self.triangular_step = triangular_step

    def __call__(self, t_cur, restart_period):
        inflection_point = self.triangular_step * restart_period
        point_of_triangle = (t_cur / inflection_point
                             if t_cur < inflection_point
                             else 1.0 - (t_cur - inflection_point)
                                  / (restart_period - inflection_point))
        return point_of_triangle


class CyclicLRWithRestarts(_LRScheduler):
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: iteration count (i.e. num_batches * num_epochs) in the first restart period
        t_mult: multiplication factor by which the next restart period will expand/shrink
        policy: ["cosine", "arccosine", "triangular", "triangular2", "exp_range"]
        min_lr: minimum allowed learning rate
        verbose: print a message on every restart
        gamma: exponent used in "exp_range" policy
        eta_on_restart_cb: callback executed on every restart, adjusts max or min lr
        eta_on_iteration_cb: callback executed on every iteration, adjusts max or min lr
        triangular_step: adjusts ratio of increasing/decreasing phases for triangular policy


    Example:
        >>> scheduler = CyclicLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    """

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2.0, last_epoch=-1, verbose=False,
                 policy="cosine", policy_fn=None, min_lr=1e-7,
                 eta_on_restart_cb=None, eta_on_iteration_cb=None,
                 gamma=1.0, triangular_step=0.5):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
                group.setdefault('minimum_lr', min_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))

        self.base_lrs = [group['initial_lr'] for group
                         in optimizer.param_groups]

        self.min_lrs = [group['minimum_lr'] for group
                        in optimizer.param_groups]

        self.base_weight_decays = [group['weight_decay'] for group
                                   in optimizer.param_groups]

        self.policy = policy
        self.eta_on_restart_cb = eta_on_restart_cb
        self.eta_on_iteration_cb = eta_on_iteration_cb
        if policy_fn is not None:
            self.policy_fn = policy_fn
        elif self.policy == "cosine":
            self.policy_fn = CosinePolicy()
        elif self.policy == "arccosine":
            self.policy_fn = ArccosinePolicy()
        elif self.policy == "triangular":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
        elif self.policy == "triangular2":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
            self.eta_on_restart_cb = ReduceMaxLROnRestart(ratio=0.5)
        elif self.policy == "exp_range":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
            self.eta_on_iteration_cb = ExpReduceMaxLROnIteration(gamma=gamma)

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.iteration = 0
        self.total_iterations = 0

        self.t_mult = t_mult
        self.verbose = verbose
        self.restart_period = math.ceil(restart_period)
        self.restarts = 0
        self.t_epoch = -1
        self.epoch = -1

        self.eta_min = 0
        self.eta_max = 1

        self.end_of_period = False
        self.batch_increments = []
        self._set_batch_increment()

    def _on_restart(self):
        if self.eta_on_restart_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_restart_cb(self.eta_min,
                                                                self.eta_max)

    def _on_iteration(self):
        if self.eta_on_iteration_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_iteration_cb(self.eta_min,
                                                                  self.eta_max,
                                                                  self.total_iterations)

    def get_lr(self, t_cur):
        eta_t = (self.eta_min + (self.eta_max - self.eta_min)
                 * self.policy_fn(t_cur, self.restart_period))

        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))

        lrs = [min_lr + (base_lr - min_lr) * eta_t for base_lr, min_lr
               in zip(self.base_lrs, self.min_lrs)]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]

        if (self.t_epoch + 1) % self.restart_period < self.t_epoch:
            self.end_of_period = True

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                print("Restart {} at epoch {}".format(self.restarts + 1,
                                                      self.last_epoch))
            self.restart_period = math.ceil(self.restart_period * self.t_mult)
            self.restarts += 1
            self.t_epoch = 0
            self._on_restart()
            self.end_of_period = False

        return zip(lrs, weight_decays)

    def _set_batch_increment(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.iteration = 0
        self.batch_increments = torch.linspace(0, 1, batches_in_epoch).tolist()

    def step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_increment()
        self.batch_step()

    def batch_step(self):
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self._on_iteration()
            self.iteration += 1
            self.total_iterations += 1
        except (IndexError):
            raise StopIteration("Epoch size and batch size used in the "
                                "training loop and while initializing "
                                "scheduler should be the same.")

        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,
                                                   self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay

