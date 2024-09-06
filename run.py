#!/usr/bin/env python3

import os
import sys

if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from roop import core

if __name__ == '__main__':
    core.run()