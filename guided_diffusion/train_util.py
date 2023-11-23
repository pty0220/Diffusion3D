import os
import blobfile as bf
import copy
import functools
import tqdm
import numpy as np
import torch as th
import sys
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

import natsort
import glob

sys.path.append(".")

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        resume_checkpoint,
        use_fp16=True,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        epochTotal,
        reportStep,
        testImage 
    ):
        self.model = model
        self.dataloader=dataloader
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.epochTotal = epochTotal
        self.reportStep = reportStep
        self.testImage = testImage
        
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.sync_cuda = th.cuda.is_available()
        self.lossStack = []
        
        self._load_and_sync_parameters()
        
        self.ema_rate = ([ema_rate] if isinstance(ema_rate, float)
                         else [float(x) for x in ema_rate.split(",")])
        
        self.mp_trainer = MixedPrecisionTrainer(model=self.model, use_fp16=self.use_fp16, fp16_scale_growth=fp16_scale_growth,)

        self.opt = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
     
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [copy.deepcopy(self.mp_trainer.master_params) for _ in range(len(self.ema_rate))]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,)
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.model
        


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step, filePath = parse_resume_step_from_filename(resume_checkpoint)
            
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {filePath}...")
                self.model.load_state_dict(dist_util.load_state_dict(filePath, map_location=dist_util.dev()))
            
            lossStack = np.load(os.path.join(resume_checkpoint, "losses.npy"))
            temp = len(lossStack)%200
            lossStack = lossStack[:-temp]
            self.lossStack = lossStack.tolist()
            a=1    
        # dist_util.sync_params(self.model.parameters())
        
    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        
        # dist_util.sync_params(ema_params)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt")
        
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        
        progressPath = os.path.join(get_blob_logdir(), 'progress.csv')
        sys.stdout = open(progressPath, 'w')
        epochTotal = self.epochTotal
        iter = 0
               
        for e in range(self.resume_step, epochTotal):
            with tqdm.tqdm(self.dataloader, dynamic_ncols=True, file=sys.stdout) as tqdmDataLoader:
                for datas in tqdmDataLoader:
                    batch = datas['A']['data']
                    cond = datas['B']['data']
                    self.run_step(batch, cond)
                    iter = iter+1
                    tqdmDataLoader.set_postfix(ordered_dict={"epoch": e, "loss: ": self.loss})
                    self.lossStack.append(self.loss.item())

                    if iter % self.reportStep == 0:
                        logger.makeLossPlot(self.lossStack, self.reportStep, e)
                        
                        batch_np = np.squeeze(batch.detach().cpu().numpy())
                        cond_np = np.squeeze(cond.detach().cpu().numpy())
                        sample_np = np.squeeze(self.sample.detach().cpu().numpy())
                        A = batch_np[0, :, :, 16]
                        B = cond_np[0, :, :, 16]
                        B_hat = sample_np[:, :, 16]

                        logger.makeTestFigure(A, B, B_hat)
                        
                                        
                self.epoch = e
                self.save()
                

    def run_step(self, batch, cond):
        batch=th.cat((batch, cond), dim=1)
        cond={}
        
        sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)

        if took_step:
            self._update_ema()
            
        self._anneal_lr()
        self.log_step()
        
        self.sample = sample
        return sample

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev()) for k, v in cond.items()}
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(self.diffusion.training_losses_A2B, self.ddp_model, micro, t, model_kwargs=micro_cond,)

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            losses = losses1[0]
            sample = losses1[1]
            loss = (losses["loss"] * weights).mean()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())
            
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

            self.loss = loss
            self.mp_trainer.backward(loss)
            
            return  sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            
            if dist.get_rank() == 0:
                if not rate:
                    # logger.log(f"saving model {(self.epoch+1)}...")
                    filename = f"model{(self.epoch+1)}.pt"
                else:
                    # logger.log(f"saving ema model {(self.epoch)}...")
                    filename = f"model{rate}_{(self.epoch)}.pt"

                with bf.BlobFile(bf.join(get_blob_logdir(), 'model', filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        
        lossStack = np.asarray(self.lossStack)
        np.save(os.path.join(get_blob_logdir(), "losses"), lossStack)
        
        # logger.sampleTest(self)
        # for rate, params in zip(self.ema_rate, self.ema_params):
        #     save_checkpoint(rate, params)

def parse_resume_step_from_filename(resume_checkpoint):
    
    modelFolder = os.path.join(resume_checkpoint, "model")
    modelFiles = glob.glob(os.path.join(modelFolder, '*.pt'))
    modelFiles = natsort.natsorted(modelFiles)   
    modelFinalFile = modelFiles[-1]
    
    temp = modelFinalFile.split("model")
    if len(temp) < 2:
        return 0
    epoch = temp[-1].split(".")[0]
    
    try:
        return int(epoch), modelFinalFile
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)