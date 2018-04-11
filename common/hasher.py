#
# Import
#
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import hashlib

def get_hash(path):
    hasher = hashlib.md5()
    with open(path, 'rb') as file:
        buf = file.read()
        hasher.update(buf) 
    return hasher.hexdigest()
