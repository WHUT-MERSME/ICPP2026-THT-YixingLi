import argparse
import time
import yaml
import json
import os
import numpy as np
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.clock_driven import functional
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.models import create_model, safe_model_name
from timm.utils import *
from timm.optim import create_optimizer_v2
import model
import math

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True

# Argument parsing
parser = argparse.ArgumentParser(description="PyTorch CIFAR10-DVS Validation")
parser.add_argument("--data-dir", default="", help="path to dataset")
parser.add_argument("--dataset", default="cifar10dvs", help="dataset type")
parser.add_argument("--model", default="THT", type=str, help="Name of model to validate")
parser.add_argument("--T", type=int, default=16, help="simulation time steps")
parser.add_argument("--patch-size", type=int, default=16, help="Image patch size")
parser.add_argument("--img-size", type=int, default=128, help="Image size")
parser.add_argument("--in-channels", type=int, default=2, help="input channels")
parser.add_argument("--resume", default="", type=str,
                    help="Resume from checkpoint")

parser.add_argument("-b", "--batch-size", type=int, default=16, help="input batch size")
parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
parser.add_argument("--amp", action="store_true", default=True, help="Use AMP mixed precision")
parser.add_argument("--output", default="./output/valid", type=str, help="path to output folder")
parser.add_argument("--experiment", default="", type=str, help="name of experiment")
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--seed", type=int, default=42, help="random seed")

_logger = logging.getLogger("valid")
stream_handler = logging.StreamHandler()
format_str = "%(asctime)s %(levelname)s: %(message)s"
stream_handler.setFormatter(logging.Formatter(format_str))
_logger.addHandler(stream_handler)
_logger.propagate = False

def _parse_args():
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def register_hooks(model, num_heads=16):
    conv_input_sizes = {}
    qkv_dims = {}
    hooks = []

    def hook_conv_fn(name, is_conv1d=False):
        def hook(module, input, output):
            if name not in conv_input_sizes:
                if is_conv1d:
                    seq_len = input[0].shape[-1]
                    conv_input_sizes[name] = seq_len
                else:
                    h_in, w_in = input[0].shape[-2], input[0].shape[-1]
                    conv_input_sizes[name] = (h_in, w_in)
        return hook

    def hook_qkv_fn(name, num_heads):
        def hook(module, input, output):
            if name not in qkv_dims:
                T_times_B, C, N = output.shape
                n = N
                d = C // num_heads
                qkv_dims[name] = (n, d)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hook = module.register_forward_hook(hook_conv_fn(name))
            hooks.append(hook)
        elif isinstance(module, nn.Conv1d) and 'q_conv' in name:
            hook = module.register_forward_hook(hook_qkv_fn(name, num_heads))
            hooks.append(hook)

    return conv_input_sizes, qkv_dims, hooks

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])
    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])
    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])
    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

def validate(model, loader, loss_fn, args, output_dir=None, amp_autocast=suppress, log_suffix=""):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    def calc_non_zero_rate(s_dict, nz_dict, idx, t):
        for k, v_ in s_dict.items():
            if t >= v_.shape[0]:
                continue
            v = v_[t, ...]
            x_shape = torch.tensor(list(v.shape))
            all_neural = torch.prod(x_shape)
            z = torch.nonzero(v)
            if k in nz_dict.keys():
                nz_dict[k] += (z.shape[0] / all_neural).item() / idx
            else:
                nz_dict[k] = (z.shape[0] / all_neural).item() / idx
        return nz_dict

    def calc_firing_rate(s_dict, fr_dict, idx, t):
        for k, v_ in s_dict.items():
            if t >= v_.shape[0]:
                continue
            v = v_[t, ...]
            mean_val = v.mean().item()
            if mean_val < 0:
                _logger.warning(f"Negative firing rate detected for {k} at t={t}: {mean_val}")
            if k in fr_dict.keys():
                fr_dict[k] += mean_val / idx
            else:
                fr_dict[k] = mean_val / idx
        return fr_dict

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    fr_dict = {f"t{i}": dict() for i in range(args.T)}
    nz_dict = {f"t{i}": dict() for i in range(args.T)}

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            hook_dict = {}
            with amp_autocast():
                output = model(input, hook=hook_dict)

            for t in range(args.T):
                fr_single_dict = calc_firing_rate(
                    hook_dict, fr_dict["t" + str(t)], last_idx, t
                )
                fr_dict["t" + str(t)] = fr_single_dict
                nz_single_dict = calc_non_zero_rate(
                    hook_dict, nz_dict["t" + str(t)], last_idx, t
                )
                nz_dict["t" + str(t)] = nz_single_dict

            output = output.mean(0)
            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (
                    last_batch or batch_idx % 100 == 0
            ):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                    )
                )

    metrics = OrderedDict(
        [
            ("loss", losses_m.avg),
            ("top1", top1_m.avg),
            ("top5", top5_m.avg),
            ("non_zero", nz_dict),
            ("firing_rate", fr_dict),
        ]
    )

    return metrics

def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            "Validation in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        _logger.info("Validation with a single process on 1 GPU.")

    use_amp = None
    if args.amp:
        if has_native_amp:
            use_amp = "native"
        elif has_apex:
            use_amp = "apex"
        else:
            _logger.warning("Neither APEX nor native Torch AMP is available, using float32.")

    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = create_model(
        args.model,
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    if args.local_rank == 0:
        _logger.info("Creating model")
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _logger.info(f"number of params: {n_parameters}")

    if args.local_rank == 0:
        conv_input_sizes, qkv_dims, hooks = register_hooks(model)

    model.cuda()
    optimizer = create_optimizer_v2(model, opt="adamw", lr=1e-3, weight_decay=0.06)
    amp_autocast = suppress
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX AMP. Validation in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info("Using native Torch AMP. Validation in mixed precision.")
    else:
        if args.local_rank == 0:
            _logger.info("AMP not enabled. Validation in float32.")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if args.local_rank == 0:
            _logger.info(f"Resumed model from checkpoint: {args.resume}")

    if args.distributed:
        model = NativeDDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    dataset_eval = CIFAR10DVS(
        args.data_dir,
        data_type="frame",
        frames_number=args.T,
        split_by="number",
    )
    _, dataset_eval = split_to_train_test_set(0.9, dataset_eval, 10)

    loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = "-".join(
            [
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                f"data-{args.dataset}",
                f"t-{args.T}",
            ]
        )
    output_dir = get_outdir(args.output if args.output else "./output/valid", exp_name)
    if args.rank == 0:
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f"{args.model}.log"), "w"
        )
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(logging.INFO)
        _logger.addHandler(file_handler)

    try:
        eval_metrics = validate(
            model,
            loader_eval,
            validate_loss_fn,
            args,
            output_dir=output_dir,
            amp_autocast=amp_autocast,
        )
        if args.local_rank == 0:
            lif_firing_rates = {}
            valid_lif_keys = [k for k in eval_metrics["firing_rate"]["t0"].keys() if "lif" in k.lower()]
            for layer in valid_lif_keys:
                avg_rate = sum(
                    eval_metrics["firing_rate"][f"t{t}"][layer] for t in range(args.T)) / args.T
                lif_firing_rates[layer] = avg_rate

            model_arch = str(model)

            conv_layers_details = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    if name in conv_input_sizes:
                        h_in, w_in = conv_input_sizes[name]
                        conv_layers_details.append(
                            f"Layer: {name}, kernel_size={kernel_size}, "
                            f"input_feature_map_size=({h_in}, {w_in}), "
                            f"in_channels={in_channels}, out_channels={out_channels}"
                        )
                    else:
                        conv_layers_details.append(
                            f"Layer: {name}, kernel_size={kernel_size}, "
                            f"input_feature_map_size=unknown, "
                            f"in_channels={in_channels}, out_channels={out_channels}"
                        )
                elif isinstance(module, nn.Conv1d):
                    kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    if name in conv_input_sizes:
                        seq_len = conv_input_sizes[name]
                        conv_layers_details.append(
                            f"Layer: {name}, kernel_size={kernel_size}, "
                            f"input_feature_map_size={seq_len}, "
                            f"in_channels={in_channels}, out_channels={out_channels}"
                        )
                    else:
                        conv_layers_details.append(
                            f"Layer: {name}, kernel_size={kernel_size}, "
                            f"input_feature_map_size=unknown, "
                            f"in_channels={in_channels}, out_channels={out_channels}"
                        )

            qkv_dims_details = []
            for name, (n, d) in qkv_dims.items():
                qkv_dims_details.append(
                    f"Layer: {name}, n={n}, d={d}"
                )

            lif_layers_details = [f"Layer: {layer}, firing_rate={rate:.4f}" for layer, rate in lif_firing_rates.items()]

            with open(os.path.join(output_dir, "model_info.txt"), "w") as f:
                f.write("Model Architecture:\n")
                f.write(model_arch + "\n\n")
                f.write("Convolutional Layers:\n")
                for detail in conv_layers_details:
                    f.write(detail + "\n")
                f.write("\nQKV Feature Map Dimensions:\n")
                for detail in qkv_dims_details:
                    f.write(detail + "\n")
                f.write("\nLIF Layers Firing Rates:\n")
                for detail in lif_layers_details:
                    f.write(detail + "\n")

            for hook in hooks:
                hook.remove()

            firing_rate_str = json.dumps(eval_metrics["firing_rate"], indent=4)
            _logger.info("firing_rate: ")
            _logger.info(firing_rate_str)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()