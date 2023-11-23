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

import time
import shutil

from sample import A2B
from multiprocessing import Process, Value, Array
import multiprocessing
import time 


checkPointPaths = ["/home/pty/Diffusion/DiffusionModel3D/checkpoint/Tanh_size256_2023-09-24-15-44-42"]

APaths = natsort.natsorted(glob.glob(os.path.join("/home/pty/Diffusion/ImageCT2MR/Testing/Normalized2/CTNor*.nii")))
# BPaths = natsort.natsorted(glob.glob(os.path.join("/home/pty/Diffusion/ImageCT2MR/Testing/Normalized2/MRNor*.nii")))

epochs = [200]
batchSize = 1

GPUnum = 0
Multi = False

################################################################
################################################################

if Multi == False:
    for cPath in checkPointPaths:
        for e in epochs:
            for APath in APaths:
                makeB = A2B(cPath, APath, batchSize=batchSize)
                makeB.time = 10 
                makeB.modelFilePath = makeB.modelFilePaths[-1]
                makeB.defineModel()
                makeB.defineGirds()
                makeB.makeB(makeB.gridSampler, 'cuda:'+str(GPUnum))
                makeB.saveCT()
                # makeCT.makePlot(CTPaths[i])

################################################################
################################################################

if __name__ == '__main__':         
    if Multi == True:
        multiprocessing.set_start_method("spawn", True)   
        for e in epochs:
            for cPath in checkPointPaths:
                while True:
                    if not os.path.exists(os.path.join(cPath, 'model', 'model'+str(e)+'.pt')):
                        print("waiting model ... ")
                        time.sleep(100)
                        
                    else:
                        print("model exist start sampling !")
                        for APath in APaths:
                            makeB = A2B(cPath, APath, batchSize=batchSize) 
                            makeB.time = 10      
                            makeB.modelFilePath = makeB.modelFilePaths[-1]
                            makeB.defineModel()
                            makeB.defineGirds()
                            finishProcess = []
                            
                            if not os.path.exists(makeB.synBPath):
                                for gpu in range(len(makeB.freeGPUs)):    
                                    proc = Process(target = makeB.makeB, args= (makeB.gridSamplerSplited[gpu], 'cuda:'+str(makeCT.freeGPUs[gpu])))
                                    proc.start()
                                    finishProcess.append(proc)
                                    
                                for f in range(len(finishProcess)):
                                    finishProcess[f].join()
                                            
                                makeB.saveCT()
                            else:
                                print("Exist :", makeB.synBPath)

