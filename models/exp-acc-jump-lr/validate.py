#
# Import
#
import sys, os
sys.path.insert(0, os.getcwd())

import tensorflow as tf
from models.base import validate


#
# Settings
#
BaseValidator = validate.BaseValidator


#
# Class
#
class Validator(BaseValidator):
    
    def __init__(self):
        BaseValidator.__init__(self)

if __name__ == '__main__':
    v = Validator()
    v.validate()

