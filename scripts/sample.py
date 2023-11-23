"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys
import random
import argparse
import os
import csv
import glob
import natsort
import tqdm

sys.path.append(".")

import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from niiCook import niiCook
from guided_diffusion import dist_util, logger
from guided_diffusion.dataLoader import  dataModule
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict, add_dict_to_argparser,)


import SimpleITK as sitk
import torchio as tio

import torch.multiprocessing as mp
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from multiprocessing import Process, Value, Array
import multiprocessing
import time 
import shutil


class A2B:
    def __init__(self, checkpointPath, APath, batchSize=1, epochNumber=-1, synBPath=True): 

        self.checkpointPath = checkpointPath
        self.modelFolderPath = os.path.join(checkpointPath, 'model')
        self.modelFilePaths = natsort.natsorted(glob.glob(os.path.join(self.modelFolderPath, "*.pt")))

        self.modelFilePath = self.modelFilePaths[epochNumber]
        self.batchSize = batchSize
        self.APath = APath
        
        self.dataSet = dataModule(None, None, 'test', patch_size=[256,256,32], batch_size=batchSize, sample_per_volume=None)
        self.dataSet.prepare_test_data(self.APath)
        self.dataSet.setup()
        self.subjectDatas = self.dataSet.test_dataloader()
        
        self.patch_overlab = 0,0,16

        self.freeGPUs = dist_util.findFreeGPUs()
        self.gridPath = os.path.join("gridParts", time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())))
        self.synBPath = synBPath    
        self.time = 1000


    def readParameter(self, path):
        data_dict = {}
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                key, value = row
                try:
                    data_dict[key] = eval(value)
                except:
                    data_dict[key] = value
                if key == 'attention_resolutions':
                    data_dict[key] = value
        return data_dict   
        
        
    def defineModel(self):
        
        if self.freeGPUs == None:
            self.freeGPUs = dist_util.findFreeGPUs()
        
        dist_util.setup_dist(None)

        argsDict = {}
        argsDictSaved = self.readParameter(glob.glob(os.path.join(self.checkpointPath, "*parameter.txt"))[-1])
        argsDict.update(model_and_diffusion_defaults())
        argsDict.update(argsDictSaved)

        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, argsDict)
        args = parser.parse_args()
        
        # ##
        # args.predict_xstart = False
        # ##
        
        self.args = args
        model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
        model.load_state_dict(dist_util.load_state_dict(self.modelFilePath, map_location="cuda"))       
        model.to(dist_util.dev())

        if args.use_fp16:
            model.convert_to_fp16()

        model.to('cuda')
        model.eval()
        
        self.model = model
        self.diffusion = diffusion
        
        epoch = self.modelFilePath.split('model')[-1].split('.pt')[0]
        model =self.checkpointPath.split('/')[-1]
        
        saveFolder = os.path.join(os.path.dirname(self.APath), model, epoch)
        os.makedirs(saveFolder, exist_ok= True)
        
        AName =  os.path.basename(os.path.normpath(self.APath))
        synBName = "3D_sB_"+model+"_epoch"+epoch+"_time_"+str(self.time)+"_from_"+AName
        
        self.synBPath = os.path.join(saveFolder, synBName)
        self.epoch = epoch                        
        
                        
    def tensorToNii(self, sample, foreground, result_fpath):
        numpy_arr = foreground.squeeze().permute(2,1,0).detach().numpy()
        #numpy_arr = (numpy_arr - numpy_arr.min())*(2000/(numpy_arr.max()-numpy_arr.min()))

        spacing = sample['A'].spacing
        origin = sample['A'].origin
        direction = sample['A'].direction

        mat_o = (-1,-1,1)
        mat_d = (-1,-1,-1,-1,-1,-1,1,1,1)
        new_direction = tuple(m*d for m, d in zip(mat_d,direction))
        new_origin = tuple(m*o for m, o in zip(mat_o,origin))

        sitk_image = sitk.GetImageFromArray(numpy_arr)
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(new_origin)
        sitk_image.SetDirection(new_direction)

        sitk.WriteImage(sitk_image, result_fpath)
        print("save B :", result_fpath)


    def makeBpatch(self, Agrid, cudaName):
        model = self.model
        model.to(cudaName)
        
        diffusion = self.diffusion

        Agrid = Agrid['A']['data']
        a = Agrid.max()
        b = Agrid.min()
        c = th.randn_like(Agrid)
        diffusionInput = th.cat((Agrid, c), dim=1)     #add a noise channel$

        synB, _, _ =  diffusion.p_sample_loop_known(model, list(diffusionInput.size()), diffusionInput, time=self.time)
        aa = synB.max()
        bb = synB.min()
        return synB
        
        
    def defineGirds(self):
        self.subject = self.subjectDatas[0]
        self.gridSampler = tio.inference.GridSampler(self.subject, self.dataSet.patch_size, patch_overlap=self.patch_overlab,)            
        self.aggregator = tio.inference.GridAggregator(self.gridSampler, overlap_mode='hann')  
        
        self.GPUallocatedNumber = self.createDivisionList(len(self.gridSampler), self.freeGPUs)
        self.gridSamplerSplited = random_split(self.gridSampler, self.GPUallocatedNumber)
        os.makedirs(self.gridPath, exist_ok=True)


    def makeB(self, gridSamplerSplited, cudaName):
        subjectGrids = DataLoader(gridSamplerSplited, batch_size=self.batchSize)
        partNum = 0
        
        for grid in tqdm.tqdm(subjectGrids):
            Bgrid = self.makeBpatch(grid, cudaName)
            locations = grid[tio.LOCATION]
            gridResult = [Bgrid, locations]
            
            gpu = cudaName.split(":")[-1]
            th.save(gridResult, os.path.join(self.gridPath, "Grid_GPU"+gpu+"_Part"+str(partNum)+".pt"))
            partNum = partNum+1
            
            
    def saveCT(self):
        paths = glob.glob(os.path.join(self.gridPath, "*.pt"))
        
        for i in range(len(paths)):
            result = th.load(paths[i])
            grid = result[0]
            locations = result[1]
            
            self.aggregator.add_batch(grid, locations=locations)
            foreground = self.aggregator.get_output_tensor()
        
        self.tensorToNii(self.subject, foreground, self.synBPath)
        shutil.rmtree(self.gridPath)


    def createDivisionList(self, number, divisor):       
        divisor = len(divisor)
        
        if divisor==0:
            divisor = 1
        
        quotient = number // divisor  
        remainder = number % divisor  
        result = [quotient] * divisor

        for i in range(remainder):
            result[i] += 1

        return result
    
    
    def makePlot(self, CTPath):
        cook = niiCook()
        cook.readSavedFile(self.APath)
        half = int(cook.array.shape[0]/2)
        A = np.squeeze(cook.array[half, :, :])
        
        cook.readSavedFile(CTPath)
        self.B = cook.array
        B = np.squeeze(cook.array[half, :, :])
        
        cook.readSavedFile(self.synCTPath)
        self.synB = cook.array
        synB = np.squeeze(cook.array[half, :, :])
        
        savePath = os.path.join(self.checkpointPath, 'result', str(self.epoch)+'.png')
        logger.makeTestFigure(A, B, synB, savePath)

