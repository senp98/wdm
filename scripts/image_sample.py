"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from wdm import dist_util, logger
from wdm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    dict_sub_dict,
)
from wdm.wdp_util import generate_wdp_trigger, weight_perturb
from scripts.plot import plot_process_demo
import yaml


def main(args, wp_all=False, dp_all=False, ds_all=False):
    sample_dir = args["model_path"].split("/")[1]
    model_name=args["model_path"].split("/")[-1]
    ts=model_name.split(".")[1]

    dist_util.setup_dist()
    if wp_all:
        ns = args["num_samples"]
        wp_std = args["weight_perturb_std"]
        logger.configure(dir=f"exps/{sample_dir}/{ns}_full_wp{wp_std}")
        logger.log(f"weight_perturb_std is {wp_std}")
    elif dp_all:
        ns = args["num_samples"]
        dp = args["dropout"]
        logger.configure(dir=f"exps/{sample_dir}/{ns}_full_dp{dp}")
        logger.log(f"dropout is {dp}")
    elif ds_all:
        ns = args["num_samples"]
        ds = args["diffusion_steps"]
        logger.configure(dir=f"exps/{sample_dir}/{ns}_full_dp{ds}")
        logger.log(f"diffusion_steps is {ds}")
    else:
        logger.configure(dir=f"exps/{sample_dir}/{ts}")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **dict_sub_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args["model_path"], bcast=True, map_location="cpu")
    )
    if args["use_fp16"]:
        model.convert_to_fp16()
    if args["weight_perturb_std"] > 0:
        res = weight_perturb(model, args["weight_perturb_std"])
        if res:
            logger.log("weight perturbation completed...")

    model.to(dist_util.dev())
    #model.eval()
    model.train()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_process_images = []
    if args["wdp_sample"]:
        wdp_gamma1 = args["wdp_gamma1"]
        wdp_trigger_kwargs = None
        if args["wdp_trigger_path"]:
            wdp_trigger_kwargs = {"wdp_trigger_path": args["wdp_trigger_path"]}

        wdp_trigger = generate_wdp_trigger(
            args["wdp_key"],
            args["wdp_trigger_type"],
            (args["batch_size"], 3, args["image_size"], args["image_size"]),
            wdp_trigger_kwargs=wdp_trigger_kwargs,
        )

    while len(all_images) * args["batch_size"] < args["num_samples"]:
        model_kwargs = {}
        if args["class_cond"]:
            classes = th.randint(
                low=0,
                high=NUM_CLASSES,
                size=(args["batch_size"]),
                device=dist_util.dev(),
            )
            model_kwargs["y"] = classes

        if args["wdp_sample"]:
            # TODO: wdp_ddim_sample_loop not implemented
            sample_fn = (
                diffusion.wdp_p_sample_loop
                if not args["use_ddim"]
                else diffusion.ddim_sample_loop
            )
            sample, process_samples = sample_fn(
                model,
                (args["batch_size"], 3, args["image_size"], args["image_size"]),
                wdp_gamma1,
                wdp_trigger,
                clip_denoised=args["clip_denoised"],
                model_kwargs=model_kwargs,
                progress=args["progress"],
                demo=args["wdp_process_demo"],
            )

        else:
            sample_fn = (
                diffusion.p_sample_loop
                if not args["use_ddim"]
                else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args["batch_size"], 3, args["image_size"], args["image_size"]),
                clip_denoised=args["clip_denoised"],
                model_kwargs=model_kwargs,
                progress=args["progress"],
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        if args["class_cond"]:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        if args["wdp_process_demo"]:
            process_samples = ((process_samples + 1) * 127.5).clamp(0, 255).to(th.uint8)
            process_samples = process_samples.permute(0, 1, 3, 4, 2)
            process_samples = process_samples.contiguous()

            gathered_process_samples = [
                th.zeros_like(process_samples) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(
                gathered_process_samples, process_samples
            )  # gather not supported with NCCL
            all_process_images.extend(
                [
                    process_samples.cpu().numpy()
                    for process_samples in gathered_process_samples
                ]
            )
        sample_num = len(all_images) * args["batch_size"]
        logger.log(f"created {sample_num} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args["num_samples"]]
    if args["class_cond"]:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args["num_samples"]]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args["class_cond"]:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
    if args["wdp_process_demo"]:
        process_arr = np.concatenate(all_process_images, axis=0)
        process_arr = process_arr[:, : args["num_samples"], :, :, :]
        out_path = os.path.join(logger.get_dir(), f"process_samples_{shape_str}.npz")
        np.savez(out_path, process_arr)
        plot_process_demo(process_arr)

    dist.barrier()
    logger.log("sampling complete")


if __name__ == "__main__":
    # th.manual_seed(3042)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, dict(p=""))
    config_path = parser.parse_args().p
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)

    model_paths = args["model_path"].split(",")

    for path in model_paths:
        args["model_path"] = path
        if args["weight_perturb_std"] == 1:
            stds = [
                # 0.002,
                # 0.004,
                # 0.006,
                # 0.008,
                0.010,
                # 0.012,
                # 0.014,
                # 0.016,
                # 0.018,
                # 0.020,
            ]
            for std in stds:
                args["weight_perturb_std"] = std
                main(args, wp_all=True)
        elif args["dropout"] == 10:
            stds = [
                0.1,
                0.2,
                0.3,
                0.4,
            ]
            for std in stds:
                args["dropout"] = std
                main(args, dp_all=True)
        elif args["diffusion_steps"] == -1:
            diffusion_steps = [
                1400,
                1200,
                800,
                600,
            ]
            for ds in diffusion_steps:
                args["diffusion_steps"] = ds
                main(args, ds_all=True)
        else:
            main(args)
