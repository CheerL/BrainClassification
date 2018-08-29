import os
import random
from collections import deque

from config import DATA_PATH, MODS, TEST_RATE, VAL_RATE
from utils.lazy_property import lazy_property


class DataManager(object):
    def __init__(self, data_path=DATA_PATH, mods=MODS):
        self.mods = mods
        self.data_path = data_path
        self.file_list = list()
        self.train_list = list()
        self.val_list = list()
        self.sitk_image = dict()
        self.sitk_label = dict()
        self.numpy_data = dict()
        self.test_rate = TEST_RATE
        self.val_rate = VAL_RATE

    @property
    def dim(self):
        if not hasattr(self, '_dim'):
            setattr(self, '_dim', 0)
        return getattr(self, '_dim', 0)

    @dim.setter
    def dim(self, val):
        if getattr(self, '_dim', 0) is not val:
            setattr(self, '_dim', val)
            if hasattr(self, '_lazy_left_dim'):
                delattr(self, '_lazy_left_dim')

    @lazy_property
    def left_dim(self):
        left_dim = set(range(3))
        left_dim.remove(self.dim)
        return tuple(left_dim)

    def create_file_list(self, limit=0):
        ''' find all directories containing images and put there name in the file_list'''
        if self.file_list:
            self.file_list.clear()

        stack = deque([self.data_path])

        while stack:
            now_path = stack.pop()
            if os.path.isdir(now_path):
                stack.extend([os.path.join(now_path, sub)
                              for sub in os.listdir(now_path)])
            else:
                if now_path.endswith('nii.gz'):
                    now_dir = os.path.dirname(now_path)
                    if now_dir not in self.file_list:
                        self.file_list.append(now_dir)

        self.check_file_list()
        if limit > 0:
            self.file_list = random.sample(self.file_list, k=limit)

    def check_file_list(self):
        for img_dir in self.file_list:
            img_str = ''.join(os.listdir(img_dir))
            for mod in self.mods:
                if mod not in img_str:
                    self.file_list.remove(img_dir)
                    break

    def load_image(self, file_list=None):
        ''' load images from image.nii'''
        raise NotImplementedError()

    def load_label(self, file_list=None):
        ''' load labels from label.nii'''
        raise NotImplementedError()

    def load_train_data(self, file_list=None):
        ''' load training data'''
        if file_list is None:
            if not (self.train_list and self.train_list):
                self.split_data()
            file_list = self.train_list

        self.load_image(file_list)
        self.load_label(file_list)

    def load_val_data(self, file_list=None):
        ''' load validation data'''
        if file_list is None:
            if not (self.train_list and self.val_list):
                self.split_data()
            file_list = self.val_list

        self.load_image(file_list)
        self.load_label(file_list)

    def split_data(self, val_rate=None):
        if not self.file_list:
            self.create_file_list()

        if val_rate is None:
            val_rate = self.val_rate

        total_num = len(self.file_list)
        val_num = int(total_num * val_rate)
        val_num_list = random.sample(range(total_num), val_num)
        self.val_list = [path for i, path in enumerate(
            self.file_list) if i in val_num_list]
        self.train_list = [path for i, path in enumerate(
            self.file_list) if i not in val_num_list]

    def get_train_numpy_data(self, file_list=None):
        self.load_train_data(file_list)
        if file_list is None:
            file_list = self.train_list
        self.numpy_data = self.get_numpy_data(file_list)

    def get_val_numpy_data(self, file_list=None):
        self.load_val_data(file_list)
        if file_list is None:
            file_list = self.val_list

        self.numpy_data = self.get_numpy_data(file_list)

    def get_numpy_data(self, file_list):
        ''' load numpy data from sitk data'''
        raise NotImplementedError()

    def clear_data(self):
        self.sitk_label.clear()
        self.sitk_image.clear()
        self.numpy_data.clear()
