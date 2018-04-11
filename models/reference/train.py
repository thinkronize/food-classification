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


if __name__ == '__main__':
    t = Trainer()
    t.train()

