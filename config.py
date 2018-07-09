import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
RESULT_PATH = os.path.join(ROOT_PATH, 'result')
TFR_PATH = os.path.join(ROOT_PATH, 'data', 'tfrecord')
LOG_PATH = os.path.join(ROOT_PATH, 'log')
MODEL_PATH = os.path.join(LOG_PATH, 'model')

CLASS_NUM = 1
SIZE = 240
MODS = ['t1']
MOD_NUM = len(MODS)

REF_SIZE = (SIZE, SIZE)
REF_ORIGIN = (-0.0, -239.0, 0.0)
REF_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
REF_SPACING = (1.0, 1.0)

BATCH_SIZE = 512
EPOCH_REPEAT_NUM = 100
SUMMARY_INTERVAL = 10
VER_BATCH_SIZE = 100

class Ref(object):
    size = REF_SIZE
    origin = REF_ORIGIN
    direction = REF_DIRECTION
    spacing = REF_SPACING


def path_init(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

path_init([DATA_PATH, RESULT_PATH, TFR_PATH, LOG_PATH, MODEL_PATH])
