import os
import time

RUN_TIME = time.ctime()[4:].replace(' ', '_').replace(':', '_')

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data', 'new_tfr')
TFR_PATH = os.path.join(DATA_PATH, 'tfrecord')
TEST_TFR_PATH = os.path.join(DATA_PATH, 'tfr_test')
TRAIN_TFR_PATH = os.path.join(DATA_PATH, 'tfr_train')
VAL_TFR_PATH = os.path.join(DATA_PATH, 'tfr_val')
LOG_PATH = os.path.join(ROOT_PATH, 'log', 'summary_%s' % RUN_TIME)

NONEMPTY_AREA_RATE = 0.15
CLASS_NUM = 2
SIZE = 240
MODS = ['t1']
MOD_NUM = len(MODS)

REF_SIZE = (SIZE, SIZE)
REF_ORIGIN = (-0.0, -239.0, 0.0)
REF_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
REF_SPACING = (1.0, 1.0)

LEARNING_RATE = 1.5e-2
BATCH_SIZE = 32
LR_DECAY_STEP = int(25000 / BATCH_SIZE)
LR_DECAY_RATE = 0.90
ADAM = False
MOMENTUM = 0.9
CONV_WEIGHT_DECAY = 1e-6
BATCH_NORM_DECAY = 0.995
BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_SCALE = True

TEST_RATE = 0.1
VAL_RATE = 0.1
WHOLE_REPEAT_NUM = 5
MIN_TUMOR_NUM = 10
MIN_CONNECT_TUMOR_NUM = 6
REPEAT_NUM = 15
SUMMARY_INTERVAL = 200
VAL_SUMMARY_INTERVAL = 200
VAL_INTERVAL = 5000 * int(32 / BATCH_SIZE)
NUM_GPU = 2
PS_TYPE = 'CPU'

DEFAULT_VERSION = 2
IS_PRO_SHORTCUT = True
BLOCK_SIZE = [5, 5, 5]

class Ref(object):
    size = REF_SIZE
    origin = REF_ORIGIN
    direction = REF_DIRECTION
    spacing = REF_SPACING
    square = REF_SIZE[0] * REF_SIZE[1]
