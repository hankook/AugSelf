import os
import logging

import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import ignite.distributed as idist

class Logger(object):

    def __init__(self, logdir, resume=None):
        self.logdir = logdir
        self.rank = idist.get_rank()

        handlers = [logging.StreamHandler(os.sys.stdout)]
        if logdir is not None and self.rank == 0:
            if resume is None:
                os.makedirs(logdir)
            handlers.append(logging.FileHandler(os.path.join(logdir, 'log.txt')))
            self.writer = SummaryWriter(log_dir=logdir)
        else:
            self.writer = None

        logging.basicConfig(format=f"[%(asctime)s ({self.rank})] %(message)s",
                            level=logging.INFO,
                            handlers=handlers)
        logging.info(' '.join(os.sys.argv))

    def log_msg(self, msg):
        if idist.get_rank() > 0:
            return
        logging.info(msg)

    def log(self, engine, global_step, print_msg=True, **kwargs):
        msg = f'[epoch {engine.state.epoch}] [iter {engine.state.iteration}]'
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is float:
                msg += f' [{k} {v:.4f}]'
            else:
                msg += f' [{k} {v}]'

            if self.writer is not None:
                self.writer.add_scalar(k, v, global_step)

        if print_msg:
            logging.info(msg)

    def save(self, engine, **kwargs):
        if idist.get_rank() > 0:
            return

        state = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.nn.parallel.DistributedDataParallel):
                v = v.module

            if hasattr(v, 'state_dict'):
                state[k] = v.state_dict()

            if type(v) is list and hasattr(v[0], 'state_dict'):
                state[k] = [x.state_dict() for x in v]

            if type(v) is dict and k == 'ss_predictor':
                state[k] = { y: x.state_dict() for y, x in v.items() }

        torch.save(state, os.path.join(self.logdir, f'ckpt-{engine.state.epoch}.pth'))

