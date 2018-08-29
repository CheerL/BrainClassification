import os
#import cv2
import random

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from config import NONEMPTY_AREA_RATE, TEST_TFR_PATH, TFR_PATH, Ref
from DM.DataManager import DataManager
from utils.tfrecord import generate_example, generate_writer

def array_signed(array, num, location):
    location_array = array[location*3:(location+1)*3]
    location_signed = 1 if abs(location_array.max()) > abs(location_array.min()) else -1
    target_array = array[num*3:(num+1)*3]
    target_signed = 1 if abs(target_array.max()) > abs(target_array.min()) else -1
    return target_array * (1 if location_signed == target_signed else -1)

class DataManagerNii(DataManager):
    def resample(self, img):
        width, height = Ref.size
        size = img.GetSize()
        origin = img.GetOrigin()
        spacing = img.GetSpacing()
        direction = img.GetDirection()
        direction = np.array(direction)
        for i in range(3):
            num = np.abs(direction[i*3:(i+1)*3]).argmax()
            if num == 0:
                num_1 = i
            elif num == 1:
                num_2 = i
            elif num == 2:
                num_3 = i

        if num_3 is 0:
            sp_w, sp_h, sp_d = spacing[num_2], spacing[num_1], spacing[num_3]
            size_w, size_h, size_d = size[num_2], size[num_1], size[num_3]
        elif num_3 is 2:
            sp_w, sp_h, sp_d = spacing[num_1], spacing[num_2], spacing[num_3]
            size_w, size_h, size_d = size[num_1], size[num_2], size[num_3]

        if (num_1, num_2, num_3) == (0, 1, 2) and (size_w, size_h) == (width, height):
            return img

        interpolator = sitk.sitkLinear
        direction = tuple(np.concatenate((
            array_signed(direction, num_1, 0),
            array_signed(direction, num_2, 1),
            array_signed(direction, num_3, 2)
        )))
        spacing = (1, 1, sp_d)
        size = (width, height, size_d)
        width_signed = 1 if sum(direction[0:3]) > 0 else -1
        height_signed = 1 if sum(direction[3:6]) > 0 else -1
        offset = (int(width_signed * (size_w * sp_w - width) / 2), int(height_signed * (size_h * sp_h - height) / 2), 0)
        trans = sitk.TranslationTransform(3)
        trans.SetOffset(offset)
        return sitk.Resample(img, size, trans, interpolator, origin, spacing, direction, 0)

    def load_image(self, file_list=None):
        if file_list is None:
            file_list = self.file_list

        for img_dir in tqdm(file_list):
            name = os.path.split(img_dir)[-1]
            for img in os.listdir(img_dir):
                mod = img.split('_')[-1].split('.')[0]
                if mod in self.mods:
                    if name not in self.sitk_image.keys():
                        self.sitk_image[name] = dict()
                    sitk_img = sitk.ReadImage(os.path.join(img_dir, img))
                    sitk_img = self.resample(sitk_img)
                    self.sitk_image[name][mod] = sitk_img

    def load_label(self, file_list=None):
        if file_list is None:
            file_list = self.file_list

        for img_dir in tqdm(file_list):
            name = os.path.split(img_dir)[-1]
            for img in os.listdir(img_dir):
                mod = img.split('_')[-1].split('.')[0]
                if mod == 'seg':
                    sitk_img = sitk.ReadImage(os.path.join(img_dir, img))
                    sitk_img = self.resample(sitk_img)
                    self.sitk_label[name] = sitk_img
                    break
            else:
                if name in self.sitk_image.keys():
                    depth = self.sitk_image[name][self.mods[0]].GetDepth()
                    self.sitk_label[name] = sitk.Image(
                        Ref.size + tuple([depth]), sitk.sitkFloat32)

    def get_numpy_data(self, file_list=None):
        numpy_data = dict()

        if file_list is None:
            file_list = self.file_list

        for img_dir in file_list:
            name = os.path.split(img_dir)[-1]
            img_dict = self.sitk_image[name]
            if name not in numpy_data.keys():
                numpy_data[name] = dict()

            numpy_data[name]['label'] = self.get_label(self.sitk_label[name])
            for mod, img in img_dict.items():
                numpy_data[name][mod] = sitk.GetArrayFromImage(
                    img).astype(np.float32)
            del self.sitk_image[name]
            del self.sitk_label[name]

        return numpy_data

    def get_label(self, data):
        numpy_data = sitk.GetArrayFromImage(data)
        # self.dim = numpy_data.shape.index(min(numpy_data.shape))
        tumor_label = numpy_data.sum(axis=self.left_dim).astype(
            np.bool).astype(np.float32)
        normal_label = 1 - tumor_label
        label = np.stack([normal_label, tumor_label], axis=1)
        return label

    def write_tfrecord(self, path=TFR_PATH, clear=True):
        if clear:
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))

        for name, data in self.numpy_data.items():
            tfr_name = os.path.join(path, '%s.tfrecord' % name)
            tfr_writer = generate_writer(tfr_name)
            for i, label in enumerate(data['label']):
                base_img = data[self.mods[0]][i]

                if len(base_img[base_img > 0]) / Ref.square > NONEMPTY_AREA_RATE:
                    img = np.stack([data[mod][i] / data[mod][i].max() for mod in self.mods], axis=2)
                    example = generate_example(img, label)
                    tfr_writer.write(example.SerializeToString())
            tfr_writer.close()

if __name__ == '__main__':
    pass
