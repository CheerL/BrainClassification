import os
import time

run_time = time.ctime()[4:].replace(' ', '_').replace(':', '_')

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
TFR_PATH = os.path.join(DATA_PATH, 'tfrecord')
TEST_TFR_PATH = os.path.join(DATA_PATH, 'tfr_test')
SUMMARY_PATH = os.path.join(ROOT_PATH, 'log', 'summary_%s' % run_time)
MODEL_PATH = os.path.join(SUMMARY_PATH, 'model')
LOG_PATH = SUMMARY_PATH

NONEMPTY_AREA_RATE = 0.1
CLASS_NUM = 2
SIZE = 240
MODS = ['t1']
MOD_NUM = len(MODS)

REF_SIZE = (SIZE, SIZE)
REF_ORIGIN = (-0.0, -239.0, 0.0)
REF_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
REF_SPACING = (1.0, 1.0)

LEARNING_RATE = 5e-2
BATCH_SIZE = 16
LR_DECAY_STEP = int(10000 / BATCH_SIZE)
LR_DECAY_RATE = 0.95
MOMENTUM = 0.9
CONV_WEIGHT_DECAY = 1e-5
BATCH_NORM_DECAY = 0.995
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_SCALE = True

TEST_RATE = 0.1
VAL_RATE = 0.1
REPEAT_NUM = 20
SUMMARY_INTERVAL = 50
VAL_INTERVAL = 500
VAL_BATCH_SIZE = 32


DEFAULT_VERSION = 2

class Ref(object):
    size = REF_SIZE
    origin = REF_ORIGIN
    direction = REF_DIRECTION
    spacing = REF_SPACING
    square = REF_SIZE[0] * REF_SIZE[1]
