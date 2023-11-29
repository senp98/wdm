"""
Train a diffusion model on images.
"""

import argparse
import numpy as np
from wdm import dist_util, logger
from wdm.image_datasets import load_data
from wdm.resample import create_named_schedule_sampler
from wdm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    dict_sub_dict,
    add_dict_to_argparser,
)
from wdm.train_util import TrainLoop
import time
import torch as th
import yaml
import os
import shutil
import torch.distributed as dist

def main(args):
    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **dict_sub_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args["schedule_sampler"], diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args["data_dir"],
        batch_size=args["batch_size"],
        image_size=args["image_size"],
        class_cond=args["class_cond"],
    )
    if args["wm_embed"]:
        wm_data = load_data(
        data_dir=args["wm_data_dir"],
        batch_size=args["wm_batch_size"],
        image_size=args["image_size"],
        class_cond=args["class_cond"],
    )
    else:
        wm_data = None
    

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args["batch_size"],
        microbatch=args["microbatch"],
        lr=args["lr"],
        ema_rate=args["ema_rate"],
        log_interval=args["log_interval"],
        save_interval=args["save_interval"],
        resume_checkpoint=args["resume_checkpoint"],
        use_fp16=args["use_fp16"],
        fp16_scale_growth=args["fp16_scale_growth"],
        schedule_sampler=schedule_sampler,
        weight_decay=args["weight_decay"],
        lr_anneal_steps=args["lr_anneal_steps"],
        # wm related
        wm_embed=args["wm_embed"],
        wm_data=wm_data,
        wm_batch_size=args["wm_batch_size"],
        wdp_key=args["wdp_key"],
        wdp_trigger_type=args["wdp_trigger_type"],
        wdp_trigger_path=args["wdp_trigger_path"],
        wdp_gamma1=args["wdp_gamma1"],
        wdp_gamma2=args["wdp_gamma2"],
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def setup_seed(seed):
     th.manual_seed(seed)
     th.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     # th.seed(seed)
     th.backends.cudnn.deterministic = True

if __name__ == "__main__":
    #setup_seed(3042)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, dict(p=""))
    config_path = parser.parse_args().p
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)
    dist_util.setup_dist()
    exp_start = time.strftime("%m-%d-%H-%M", time.localtime())
    exp_dir = f"./exps/exp_{exp_start}"
    
    if dist.get_rank() == 0:
        if not os.path.exists(exp_dir):
             os.mkdir(exp_dir)
             logger.configure(dir=f"{exp_dir}/logs")
             shutil.copy(config_path, f"{exp_dir}/train_{exp_start}.yaml")
             main(args)
        else:
            print("experiment directory exists!")
    else:
        time.sleep(5)
        main(args)
        
       

    
    

    
