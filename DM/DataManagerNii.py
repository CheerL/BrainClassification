import os

#import cv2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from DM.DataManager import DataManager
from DM.tfrecord import generate_example, generate_writer
from config import SIZE, TFR_PATH

# from pathos.multiprocessing import ProcessingPool as Pool
# print('version 5')

class RefImg(object):
    def __init__(self):
        self.size = (SIZE, SIZE)
        self.origin = (-0.0, -239.0, 0.0)
        self.direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.spacing = (1.0, 1.0)


class DataManagerNii(DataManager):
    ref = RefImg()

    def resample(self, img):
        width, height = self.ref.size
        origin = self.ref.origin
        direction = self.ref.direction

        sp_w, sp_h, sp_d = img.GetSpacing()
        img_width, img_height, img_depth = img.GetSize()

        offset = ((width - img_width * sp_w) / 2,
                  (height - img_height * sp_h) / 2, 0)
        trans = sitk.TranslationTransform(3)
        trans.SetOffset(offset)

        size = (width, height, img_depth)
        spacing = (1.0, 1.0, sp_d)
        interpolator = sitk.sitkLinear

        return sitk.Resample(img, size, trans, interpolator, origin, spacing, direction, 0)

    def need_resample(self, img):
        return (
            img.GetSize()[:2],
            img.GetSpacing()[:2],
        ) != (
            self.ref.size,
            self.ref.spacing,
        )

    def loadImage(self, fileList=None):
        if fileList is None:
            fileList = self.fileList

        for img_dir in tqdm(fileList):
            name = os.path.split(img_dir)[-1]
            for img in os.listdir(img_dir):
                mod = img.split('_')[-1].split('.')[0]
                if mod in self.mods:
                    if name not in self.sitkImage.keys():
                        self.sitkImage[name] = dict()
                    sitk_img = sitk.ReadImage(os.path.join(img_dir, img))
                    if self.need_resample(sitk_img):
                        print('img' + name)
                        sitk_img = self.resample(sitk_img)
                    self.sitkImage[name][mod] = sitk_img

    def loadGT(self, fileList=None):
        if fileList is None:
            fileList = self.fileList

        for img_dir in tqdm(fileList):
            name = os.path.split(img_dir)[-1]
            for img in os.listdir(img_dir):
                mod = img.split('_')[-1].split('.')[0]
                if mod == 'seg':
                    sitk_img = sitk.ReadImage(os.path.join(img_dir, img))
                    if self.need_resample(sitk_img):
                        print('gt' + name)
                        sitk_img = self.resample(sitk_img)
                    self.sitkGT[name] = sitk_img
                    break
            else:
                if name in self.sitkImage.keys():
                    depth = self.sitkImage[name][self.mods[0]].GetDepth()
                    self.sitkGT[name] = sitk.Image(self.ref.size + tuple([depth]), sitk.sitkFloat32)

    def getNumpyData(self, file_list, method):
        numpy_data = dict()
        for img_dir in file_list:
            name = os.path.split(img_dir)[-1]
            img_dict = self.sitkImage[name]
            if name not in numpy_data.keys():
                numpy_data[name] = dict()

            print(name)
            numpy_data[name]['label'] = self.get_label(self.sitkGT[name])
            for mod, img in img_dict.items():
                numpy_data[name][mod] = sitk.GetArrayFromImage(img).astype(dtype=np.float32)

        return numpy_data

    def get_label(self, data):
        numpy_data = sitk.GetArrayFromImage(data)
        # self.dim = numpy_data.shape.index(min(numpy_data.shape))
        label = numpy_data.sum(axis=self.left_dim).astype(np.bool).astype(np.float32)
        return label

    def write_tfrecord(self):
        for name, data in self.numpyData.items():
            tfr_name = os.path.join(TFR_PATH, '%s.tfrecord' % name)
            tfr_writer = generate_writer(tfr_name)
            for i, label in enumerate(data['label']):
                img = np.stack([data[mod][i] for mod in self.mods], axis=2)
                example = generate_example(img, label)
                tfr_writer.write(example.SerializeToString())
            tfr_writer.close()

if __name__ == '__main__':
    dm = DataManagerNii('data', 'result', None)
