"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist
import nvidia_smi

import numpy as np

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3



def setup_dist(gpu_num):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    if gpu_num != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        
    # else:
    #     freeGPU = findFreeGPU()
    #     b = ",".join(str(gpu) for gpu in freeGPU)
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in freeGPU)

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    n_gpus = th.cuda.device_count()

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpigetrank=0
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def findFreeGPUs():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    freeGPU = []
    
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        feeMemory = 100*info.free/info.total
    
        if feeMemory>90:
            freeGPU.append(i)
    
    try:
        # freeGPU.remove(7) ########??
        # freeGPU.remove(6)
        # freeGPU.remove(5)
        # freeGPU.remove(4)
        # freeGPU.remove(7)
        # freeGPU.remove(4)
        a=1
    except:
        a=1
    # freeGPU.append(-1)
            
    nvidia_smi.nvmlShutdown()
    
    return freeGPU