#
# Import
#
import sys, os
sys.path.insert(0, os.getcwd())

import tensorflow as tf
from models.base import train


#
# Settings
#
BaseTrainer = train.BaseTrainer


#
# Class
# 
class Trainer(BaseTrainer):

    def __init__(self):
        BaseTrainer.__init__(self)

    def define_optimizer(self):
        lr_val = 0.001
        lr = tf.placeholder_with_default(lr_val, shape=[])
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        return self.set_tensors({'optimizer':optimizer, 'lr':lr})

if __name__ == '__main__':
    t = Trainer()
    t.train()

