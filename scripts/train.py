import sys
import argparse
import torch as th
import json
import os 
import numpy as np

sys.path.append("..")
sys.path.append(".")

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.dataLoader import  dataModule
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict, add_dict_to_argparser,)
from guided_diffusion.train_util import TrainLoop


def main(GPU_NUMBER):
    args = create_argparser().parse_args()
    dist_util.setup_dist(GPU_NUMBER)
    
    logger.configure(args)
    logger.saveArgs(args)    

    logger.log("1. Create model and diffusion")
    model, diffusion = create_model_and_diffusion( **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("2. Define data set")
    dataSet = dataModule(args.data_dir, args.test_dir, 'train', patch_size=[256,256,64], batch_size=10, sample_per_volume= 200)
    dataSet.prepare_data()
    dataSet.setup()
    dataLoader = dataSet.train_dataloader()    
    data = iter(dataLoader)
    
    logger.log("3. Start train loop")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        dataloader=dataLoader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        epochTotal = args.epochTotal,
        reportStep = args.reportStep,
        testImage = None
    ).run_loop()

def create_argparser():
    para = dict(
        # data_dir="/home/pty/Diffusion/ImageCT2MR/Training/Normalized2",
        # test_dir = "/home/pty/Diffusion/ImageCT2MR/Training/Normalized2",
        
        data_dir="/home/pty/Diffusion/iDB-CERMEP-MRXFDG_MRI_CT/normalization_new/Train",
        test_dir = "/home/pty/Diffusion/iDB-CERMEP-MRXFDG_MRI_CT/normalization_new/Train",
        
        schedule_sampler="uniform",
        noise_schedule = "linear",
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        epochTotal = 200,
        resume_checkpoint= None,#'/home/pty/DiffusionModel3D_CT2MR/checkpoint/Tanh_size256_2023-09-24-15-44-42',#/model175.pt',#'"./results/pretrainedmodel.pt",
        use_fp16=True,
        fp16_scale_growth=1e-7,
        reportStep = 50,
        learn_sigma = False,
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        diffusion_steps = 1000,
        predict_xstart = True,
        use_new_attention_order = False,
)

    defaults = model_and_diffusion_defaults()
    defaults.update(para)
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    GPU_NUMBER = 0
    main(GPU_NUMBER)
