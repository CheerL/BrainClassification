import numpy as np
import SimpleITK as sitk
#import cv2
import random
import copy
import math
import os
from tqdm import tqdm
from collections import deque


class DataManager(object):
    def __init__(self, srcFolder, resultsDir, parameters, probabilityMap=False, mods=['t1']):
        self.mods = mods
        self.params = parameters
        self.srcFolder = srcFolder.strip('/').strip('\\').strip()
        self.resultsDir = resultsDir.strip('/').strip('\\').strip()
        self.probabilityMap = probabilityMap
        self.fileList = list()
        self.trainList = list()
        self.testList = list()
        self.sitkImage = dict()
        self.sitkGT = dict()
        self.test_rate = 0.1
        self.numpyImage = 0
        self.numpyGT = 0
        self.meanIntensityTrain = None
        self.originalSizes = dict()

        self.createFileList()

    def createFileList(self):
        ''' find all directories containing images and put there name in the fileList'''
        if self.fileList:
            self.fileList.clear()

        stack = deque([self.srcFolder])

        while stack:
            now_path = stack.pop()
            if os.path.isdir(now_path):
                stack.extend([os.path.join(now_path, sub) for sub in os.listdir(now_path)])
            else:
                if now_path.endswith('nii.gz'):
                    now_dir = os.path.dirname(now_path)
                    if now_dir not in self.fileList:
                        self.fileList.append(now_dir)
        self.checkFileList()

    def checkFileList(self):
        for img_dir in self.fileList:
            img_str = ''.join(os.listdir(img_dir))
            for mod in self.mods + ['seg']:
                if mod not in img_str:
                    self.fileList.remove(img_dir)
                    break

    def loadImage(self, fileList=None):
        ''' load images from image.nii'''
        raise NotImplementedError()


    def loadGT(self, fileList=None):
        ''' load labels from label.nii'''
        raise NotImplementedError()

    def loadTrainingData(self):
        ''' load training data'''
        if not (self.trainList and self.testList):
            self.splitData()

        self.loadImage(self.trainList)
        self.loadGT(self.trainList)

    def loadTestData(self):
        ''' load testing data or validation data'''
        if not (self.trainList and self.testList):
            self.splitData()

        self.loadImage(self.testList)
        self.loadGT(self.testList)

    def splitData(self):
        total_num = len(self.fileList)
        test_num = int(total_num * self.test_rate)
        test_num_list = random.sample(range(total_num), test_num)
        self.testList = [path for i, path in enumerate(self.fileList) if i in test_num_list]
        self.trainList = [path for i, path in enumerate(self.fileList) if i not in test_num_list]

    def getNumpyImage(self):
        self.numpyImage =  self.getNumpyData(self.sitkImage, sitk.sitkLinear)

    def getNumpyGT(self):
        self.numpyGT = self.getNumpyData(self.sitkGT, sitk.sitkLinear)

    def getNumpyData(self, dat, method):
        ''' load numpy data from sitk data'''
        raise NotImplementedError()

    def writeResultsFromNumpyLabel(self, result, key):
        ''' save the segmentation results to the result directory'''
        # toWrite=sitk.Image(img.GetSize()[0],img.GetSize()[1],img.GetSize()[2],sitk.sitkFloat32)

        # resize to the original size
        #result = skimage.transform.resize(result, self.originalSizes[key], order=3, mode='reflect', preserve_range=True)
        result = np.transpose(result, [2, 1, 0])

        if self.probabilityMap:
            result = result * 255
        else:
            result = result>0.5
            result = result.astype(np.uint8)
        toWrite = sitk.GetImageFromArray(result)

        toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()
        filename, ext = os.path.splitext(key)
        # #print join(self.resultsDir, filename + '_result' + ext)
        writer.SetFileName(os.path.join(self.resultsDir, filename + '_result.nii'))
        writer.Execute(toWrite)
