import os
import time

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
TFR_PATH = os.path.join(DATA_PATH, 'tfrecord')
TEST_TFR_PATH = os.path.join(DATA_PATH, 'tfr_test')
LOG_PATH = os.path.join(ROOT_PATH, 'log')
SUMMARY_PATH = os.path.join(LOG_PATH, 'summary_%s' % time.ctime())
MODEL_PATH = os.path.join(SUMMARY_PATH, 'model')

CLASS_NUM = 2
SIZE = 240
MODS = ['t1']
MOD_NUM = len(MODS)

REF_SIZE = (SIZE, SIZE)
REF_ORIGIN = (-0.0, -239.0, 0.0)
REF_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
REF_SPACING = (1.0, 1.0)

LEARNING_RATE = 1e-1
LR_DECAY_STEP = 400
LR_DECAY_RATE = 0.9
MOMENTUM = 0.9
BATCH_SIZE = 16
CONV_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_SCALE = True

TEST_RATE = 0.1
EPOCH_REPEAT_NUM = 3
SUMMARY_INTERVAL = 20
VER_BATCH_SIZE = 32


DEFAULT_VERSION = 2

class Ref(object):
    size = REF_SIZE
    origin = REF_ORIGIN
    direction = REF_DIRECTION
    spacing = REF_SPACING


def path_init(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

path_init([DATA_PATH, TFR_PATH, TEST_TFR_PATH, LOG_PATH])
