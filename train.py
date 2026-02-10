# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
"""
import os
import sys
_root = os.path.dirname(os.path.abspath(__file__))
# 使用 xdimo 作为模块根路径，使 from models/datasets/diffusion/utils 能正确解析到 xdimo 下
_src = os.path.join(_root, "xdimo")
if _src not in sys.path:
    sys.path.insert(0, _src)

# import wandb  # 已注释，避免 wandb 相关问题
import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os
import math
import argparse

import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from datetime import datetime
from einops import rearrange
from models import get_models
from datasets import get_dataset
from models.clip import TextEmbedder
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard, 
                   write_tensorboard, setup_distributed,
                   get_experiment_dir, text_preprocessing)
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer

#################################################################################
#                                  MFU 测算                                     #
#################################################################################

# 超算队列 GPU -> FP16 峰值 TFLOPS（gpu_6000ada / gpu_4090 / gpu_h100 / gpu_pro6000）
# 运行时根据 torch.cuda.get_device_name() 自主检测，改 -q 即可
_GPU_PEAK_TFLOPS_FP16 = [
    ("H100", 989), ("H200", 989),
    ("RTX PRO 6000", 200), ("PRO 6000", 200), ("PRO6000", 200),
    ("RTX 6000 ADA", 91.1), ("6000 ADA", 91.1),
    ("RTX 4090", 82.6),
]


def get_peak_tflops_fp16(device):
    """根据 GPU 型号返回 FP16 峰值 TFLOPS，支持三队列自主检测。"""
    try:
        name = (torch.cuda.get_device_name(device) or "").upper()
        for keyword, tflops in _GPU_PEAK_TFLOPS_FP16:
            if keyword in name:
                return tflops
    except Exception:
        pass
    return 82.6  # 未知 GPU 保守估计（按 4090）


def get_gpu_short_tag(device):
    """用于 output 目录命名：根据 GPU 型号返回短标签（如 6000ada / h200 / pro6000）。"""
    try:
        name = (torch.cuda.get_device_name(device) or "").upper()
        if "H200" in name or "H100" in name:
            return "h200"
        if "6000 ADA" in name:
            return "6000ada"
        if "PRO 6000" in name or "PRO6000" in name:
            return "pro6000"
        if "4090" in name:
            return "4090"
    except Exception:
        pass
    return "gpu"

def estimate_latte_flops_per_forward(num_frames, latent_size, hidden_size=384, num_layers=12):
    """Latte 单样本单次前向的 FLOPs 估计（Transformer: attention + MLP）。"""
    n = num_frames * (latent_size ** 2)
    d = hidden_size
    # 每层: attention 4nd^2 + 2n^2d, MLP ~8nd^2
    flops_per_layer = (4 * n * d * d + 2 * n * n * d) + 8 * n * d * d
    return num_layers * flops_per_layer


def main(args):
    # 将相对路径转为基于仓库根目录的绝对路径，便于在任意 cwd 下运行
    _root = os.path.dirname(os.path.abspath(__file__))
    for attr in ('pretrained_model_path', 'data_path', 'results_dir'):
        v = getattr(args, attr, None)
        if v and not os.path.isabs(v):
            setattr(args, attr, os.path.normpath(os.path.join(_root, v)))

    # Setup DDP:
    setup_distributed()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    #set seed 
    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    world_size = dist.get_world_size()  # 卡数，DDP 下即 GPU 数量
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
        gpu_tag = get_gpu_short_tag(device)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}-{gpu_tag}-{world_size}gpu-{timestamp}"
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
        # wandb init（已注释）
        # wandb.init(project="Latte-Training", name=f"{args.model}-{args.dataset}", mode="offline")
        # config_dict = OmegaConf.to_container(args, resolve=True)
        # wandb.config.update(config_dict)
    else:
        experiment_dir = None
        logger = create_logger(None)
        tb_writer = None
    
    #广播目录 
    experiment_dir_list = [experiment_dir]
    dist.broadcast_object_list(experiment_dir_list, src=0)
    experiment_dir = experiment_dir_list[0]
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    
    #model creation 
    model = get_models(args) #我不需要去find_model,因为我直接创建model
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)

    # # use pretrained model?
    print(f"args.pretrained is {args.pretrained}")
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        print(f"checkpoint is {checkpoint}")
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))
    else:
        print("本来就是从头训练!")
    if args.use_compile:
        model = torch.compile(model)

    # Note that parameter initialization is done within the Latte constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
  
    # set distributed training
    # 模型存在部分参数未参与 loss 的情况（如 dropout/条件分支），必须设为 True 否则 DDP 会报错
    model = DDP(model.to(device), device_ids=[local_rank], find_unused_parameters=True)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {num_params:,}")
    # MFU 测算：用当前模型的 hidden_size、depth 估计单样本前向 FLOPs（否则会沿用默认 S/2 维度导致 MFU 偏低）
    _hidden = getattr(model.module, 'hidden_size', 384)
    _depth = len(getattr(model.module, 'blocks', [])) or 12
    flops_forward = estimate_latte_flops_per_forward(
        args.num_frames, args.latent_size, hidden_size=_hidden, num_layers=_depth
    )
    peak_tflops = get_peak_tflops_fp16(device)
    world_size = dist.get_world_size()
    global_batch = args.local_batch_size * world_size
    flops_per_step = 3.0 * flops_forward * global_batch  # 前向+反向约 3x 前向
    if rank == 0:
        try:
            gpu_name = torch.cuda.get_device_name(device)
        except Exception:
            gpu_name = "unknown"
        logger.info("---------- Compute & MFU ----------")
        logger.info(f"  FLOPs 使用维度:   hidden_size={_hidden}, num_layers={_depth}")
        logger.info(f"  Parameters:        {num_params:,} ({num_params/1e6:.2f}M)")
        logger.info(f"  FLOPs/sample(fwd): {flops_forward/1e12:.4f} TFLOPs ({flops_forward/1e15:.4f} PFLOPs)")
        logger.info(f"  FLOPs/step(3×fwd×B): {flops_per_step/1e12:.4f} TFLOPs (global_batch={global_batch})")
        logger.info(f"  GPU:              {gpu_name} (peak FP16: {peak_tflops} TFLOPs/GPU)")
        logger.info(f"  MFU formula:      achieved_TFLOPs / (world_size × peak_TFLOPs), achieved = FLOPs/step × steps_per_sec")
        logger.info("-----------------------------------")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # Setup data:
    dataset = get_dataset(args)

    sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=rank,
    shuffle=True,
    seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 修改后的检查点恢复逻辑
    resume_step = 0
    if args.resume_from_checkpoint:
        if os.path.exists(checkpoint_dir):
            dirs = os.listdir(checkpoint_dir)
            dirs = [d for d in dirs if d.endswith(".pt")]
            
            if len(dirs) > 0:
                dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
                path = dirs[-1]
                logger.info(f"Resuming from checkpoint {path}")
                checkpoint_path = os.path.join(checkpoint_dir, path)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # 加载模型状态
                model.module.load_state_dict(checkpoint["model"])
                ema.load_state_dict(checkpoint["ema"])
                
                train_steps = int(path.split(".")[0])
                first_epoch = train_steps // num_update_steps_per_epoch
                resume_step = train_steps % num_update_steps_per_epoch
            else:
                logger.warning(f"No checkpoints found in {checkpoint_dir}")
        else:
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")

    if args.pretrained:
        train_steps = int(args.pretrained.split("/")[-1].split('.')[0])
    #702/64 = 11 steps/epoch
    epoch_nums = 7102 #1000000x 5 / 702 = 7102 origin_epochs 
    # for epoch in range(first_epoch, num_train_epochs):
    for epoch in range(first_epoch, epoch_nums):
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(loader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            x = video_data['video'].to(device, non_blocking=True)
            video_name = video_data['video_name']
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                b, _, _, _, _ = x.shape
                x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()

            if args.extras == 78: # text-to-video
                raise 'T2V training are Not supported at this moment!'
            elif args.extras == 2:
                model_kwargs = dict(y=video_name)
            else:
                model_kwargs = dict(y=None)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
            loss.backward()

            if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

            
            lr_scheduler.step()
            if train_steps % args.gradient_accumulation_steps == 0 and train_steps > 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                achieved_tflops = flops_per_step * steps_per_sec / 1e12
                mfu = min(achieved_tflops / (world_size * peak_tflops), 1.0)  # 防止峰值估计偏小导致 >100%
                samples_per_sec = steps_per_sec * global_batch  # 样本吞吐（samples/s），即 token/视频 吞吐
                logger.info(
                    f"(step={train_steps:07d}/epoch={epoch:04d}) Loss: {avg_loss:.4f} | GradNorm: {gradient_norm:.4f} | "
                    f"Steps/Sec: {steps_per_sec:.2f} | Samples/s: {samples_per_sec:.2f} | Achieved: {achieved_tflops:.2f} TFLOPs/s | MFU: {mfu*100:.2f}%"
                )
                write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                write_tensorboard(tb_writer, 'Samples/s', samples_per_sec, train_steps)
                write_tensorboard(tb_writer, 'MFU', mfu, train_steps)
                # wandb 记录（已注释）
                # if rank == 0:
                #     wandb.log({"Train Loss": avg_loss, "Gradient Norm": gradient_norm, "Train Steps/Sec": steps_per_sec}, step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save Latte checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        # "opt": opt.state_dict(),
                        # "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    # wandb.save(checkpoint_path)  # 已注释
                dist.barrier()

            # 达到最大步数即结束训练
            if train_steps >= args.max_train_steps:
                if rank == 0:
                    logger.info(f"Reached max_train_steps={args.max_train_steps}, stopping.")
                break
        if train_steps >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    #TODO 我必须要在这里写一个 测试 了，直接生成10个视频测FVD即可

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train Latte with the hyperparameters we used in our paper (except training iters).
    _root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ffs/ffs_train.yaml")
    args = parser.parse_args()
    config_path = args.config if os.path.isabs(args.config) else os.path.normpath(os.path.join(_root, args.config))
    main(OmegaConf.load(config_path))
