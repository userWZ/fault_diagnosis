"""
    用于记录与打印的工具函数
"""

from tensorboardX import SummaryWriter
import os
import torch


class Recoder:
    def __init__(self):
        self.metrics = {}

    def record(self, name, value):
        if name in self.metrics.keys():
            self.metrics[name].append(value)
        else:
            self.metrics[name] = [value]

    def summary(self):
        kvs = {}
        for key in self.metrics.keys():
            kvs[key] = sum(self.metrics[key]) / len(self.metrics[key])
            del self.metrics[key][:]
            self.metrics[key] = []
        return kvs


class Logger:
    def __init__(self, args):
        log_prefix = 'group_' + args.data_group + '_num_' + args.data_num + '_seed_' + str(args.seed)
        log_dir = os.path.join(args.result_dir, 'log', log_prefix)
        if not os.path.exists(log_dir):
            os.system('mkdir -p ' + log_dir)
        self.writer = SummaryWriter(log_dir)
        self.recoder = Recoder()
        self.model_dir = args.model_dir

    def record_scalar(self, name, value):
        self.recoder.record(name, value)

    def summary(self, epoch):
        kvs = self.recoder.summary()
        for key in kvs.keys():
            self.writer.add_scalar(key, kvs[key], epoch)
