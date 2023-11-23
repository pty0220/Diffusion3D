import time

import torch
from torch.utils.data import DataLoader
import monai
# import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import os
import glob
import SimpleITK as sitk

class dataModule(pl.LightningDataModule):
    def __init__(self, dir_train, dir_test, task, patch_size , batch_size, sample_per_volume):
        super().__init__()
        self.task = task
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.sample_per_volume = sample_per_volume
        self.subjects = []
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.dir_train = dir_train
        self.dir_test = dir_test

    
    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)
    
    def download_data(self):
        # self.dir_train = dir_path
        # self.dir_test = dir_path

        def getniipaths(dir_path):
            A_pathname = os.path.join(dir_path, 'CT*.nii')
            B_pathname = os.path.join(dir_path, 'MR*.nii')
            MASK_pathname = os.path.join(dir_path, 'Mask_*.nii')

            A_paths = sorted(glob.glob(A_pathname))
            B_paths = sorted(glob.glob(B_pathname))
            MASK_paths = sorted(glob.glob(MASK_pathname))

            return A_paths, B_paths, MASK_paths
        
        def getniiBpaths(dir_path):
            B_pathname = os.path.join(dir_path, 'MR*.nii')
            B_paths = sorted(glob.glob(B_pathname))

            return B_paths
        
        
        if self.task == 'train':
            A_paths, B_paths, MASK_paths = getniipaths(self.dir_train)
        elif self.task == 'test':
            A_paths = None
            MASK_paths = None
            B_paths,  = getniiBpaths(self.dir_test)

        return A_paths, B_paths, MASK_paths

    def prepare_data(self):
        A_paths, B_paths, MASK_paths = self.download_data()
        if self.task == 'train':
            for A_path, B_path, MASK_path in zip(A_paths, B_paths, MASK_paths):

                subject = tio.Subject(
                    A=tio.ScalarImage(A_path),
                    B=tio.ScalarImage(B_path),
                    MASK=tio.Image(MASK_path, type=tio.SAMPLING_MAP),
                )
                self.subjects.append(subject)

    def prepare_test_data(self, A_path):
        if self.task == 'test':
            subject = tio.Subject(A=tio.ScalarImage(A_path))
            self.subjects.append(subject)
        
    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            # tio.RescaleIntensity((-1, 1)),
            # tio.CropOrPad(self.get_max_shape(self.subjects + self.test_subjects)),
            # tio.EnsureShapeMultiple(8),  # for the U-Net
            # tio.OneHot(),
        ])
        return preprocess
    
    def get_augmentation_transform(self):
        augment = tio.Compose([
            # tio.RandomAffine(scales=(0.9,1.1), degrees=10),
            # tio.RandomGamma(p=0.5),
            # tio.RandomNoise(p=0.5),
            # tio.RandomMotion(p=0.1),
            # tio.RandomBiasField(p=0.25),
        ])
        return augment

    def setup(self):

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

        queue_length = 300
        samples_per_volume = self.sample_per_volume

        if self.task == 'train':
            train_set = tio.SubjectsDataset(self.subjects, transform=self.transform)
            sampler = tio.data.WeightedSampler(self.patch_size, 'MASK')
            self.train_patches_queue = tio.Queue(train_set, queue_length, samples_per_volume, sampler,num_workers=0)
        elif self.task == 'test':
            self.test_set = tio.SubjectsDataset(self.subjects, transform=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_patches_queue, self.batch_size)

    def test_dataloader(self):
        return self.test_set #DataLoader(self.test_set, self.batch_size)