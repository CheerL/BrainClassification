from config import *
import os
import shutil

def path_init(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def clear_empty_log():
    log_base_path = os.path.join(ROOT_PATH, 'log')
    for dir_name in os.listdir(log_base_path):
        sub_path = os.path.join(log_base_path, dir_name)
        if sub_path != LOG_PATH and os.path.isdir(sub_path):
            paths = os.listdir(sub_path)
            if not (paths and 'model' in paths):
                # os.removedirs(sub_path)
                shutil.rmtree(sub_path)

def config_save():
    config_path = os.path.join(ROOT_PATH, 'config.py')
    config_copy_path = os.path.join(LOG_PATH, 'config')
    shutil.copy(config_path, config_copy_path)

path_init([DATA_PATH, TFR_PATH, TEST_TFR_PATH, LOG_PATH])
clear_empty_log()
config_save()
