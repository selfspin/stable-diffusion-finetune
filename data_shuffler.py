from torch.utils.data import dataloader
from dataset.dataset import *
import matplotlib.pyplot as plt
from warmup_scheduler.scheduler import GradualWarmupScheduler

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import numpy as np
import time
import argparse
import os
import random
import torch.backends.cudnn
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable
import pickle
from torch.optim import lr_scheduler

from dataset.clip_image import *

import torch.distributed as dist
import torch.multiprocessing as mp
from distributed_utils import reduce_value, is_main_process


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion',
                        help='choose from [stable-diffusion, stable-diffusion-v2, stable-diffusion-v2base, clip]')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--device', type=str, default='cuda', help="training device")
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--epochs', type=int, default=100, help="training epoch")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--lr_fine', type=float, default=5e-4, help="initial learning rate in fine stage")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--ckpt_fine', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=512,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64,
                        help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32,
                        help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo', action='store_true',
                        help="only use albedo shading to train, overrides --albedo_iters")
    parser.add_argument('--albedo_iters', type=int, default=0, help="training iters that only use albedo shading")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5,
                        help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--weight_choice', type=int, default=0, help="choice for w(t)")

    # gpus
    parser.add_argument('--local_rank', type=int, default=0, help="node rank for distributed training")
    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7", help="devices: '0,1,2,3' ")
    parser.add_argument('--bs', type=str, default=16, help="per gpu batch size")
    parser.add_argument('--port', type=int, default=15684, help="port, arbitrary number in 0~65536")

    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4,
                        help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='grid', help="nerf backbone, choose from [grid, vanilla]")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--res_fine', type=int, default=512, help="image resolution in fine stage")
    parser.add_argument('--tet_res', type=int, default=256, help="resolution for tetrahedron grid")

    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5],
                        help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument('--focal_range', type=float, nargs='*', default=[0.7, 1.35], help="training camera focal range")
    parser.add_argument('--focal_range_fine', type=float, nargs='*', default=[1.2, 1.8],
                        help="training camera focal range in fine stage")
    parser.add_argument('--dir_text', action='store_true',
                        help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=90,
                        help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    parser.add_argument('--lambda_entropy', type=float, default=1e-4, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_smooth', type=float, default=0, help="loss scale for surface smoothness")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60,
                        help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    # others
    parser.add_argument('--clip-norm', default=True, action='store_true')
    parser.add_argument('--lr-max', default=0.0001, type=float)
    parser.add_argument('--towards', default='side2', type=str)

    opt = parser.parse_args()
    return opt


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(backend="nccl",
                            init_method=f'tcp://127.0.0.1:{args.port}',
                            world_size=nprocs,
                            rank=local_rank)
    args.world_size = dist.get_world_size()
    args.global_rank = dist.get_rank()
    args.bs = int(args.bs)
    seed_everything(args.seed + args.global_rank)

    if args.global_rank == 0:
        print(args)

    device = torch.device('cuda', args.local_rank)

    if is_main_process():
        print("Loading training data ...")
    train_dataset = ViewDataset(train=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset)
    trainloader = dataloader.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=False, sampler=sampler)

    test_dataset = ViewDataset(train=False)
    sampler_test = torch.utils.data.distributed.DistributedSampler(dataset=test_dataset)
    testloader = dataloader.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, sampler=sampler_test)
    if is_main_process():
        print("Loaded!")

    for max_lr in [1e-2]:
        os.makedirs(os.path.join("shuffler_checkpoints/", 'iter0', str(max_lr)), exist_ok=True)
        args.lr_max = max_lr

        parameter_setting_list = []
        model = CLIP(args.device, args)
        for name, parameter in model.named_parameters():
            if 'clip_model' in name:
                # parameter.requires_grad = False
                parameter_setting_list.append({'params': parameter, 'lr': 0.0001 * args.lr_max / 10})
            else:
                parameter_setting_list.append({'params': parameter, 'lr': args.lr_max / 10})

        if args.world_size > 1:
            process_group = torch.distributed.new_group(list(range(dist.get_world_size())))
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                              find_unused_parameters=True)

        # opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_max / 10, weight_decay=0.001)
        opt = optim.Adam(parameter_setting_list, lr=args.lr_max / 10, weight_decay=0.001)
        scaler = torch.cuda.amp.GradScaler()
        # lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs],
        #                                   [0, args.lr_max, args.lr_max / 20.0, 0])[0]
        Cosinescheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        scheduler = GradualWarmupScheduler(opt, multiplier=10, total_epoch=args.epochs / 10,
                                           after_scheduler=Cosinescheduler)
        criterion = DiffusionLoss()
        loss_list = []
        lr_list = []

        for epoch in range(args.epochs):
            if args.world_size > 1:
                sampler.set_epoch(epoch)
            start = time.time()
            train_loss, train_acc, n, loss_value, lr = 0, 0, 0, 0, 0
            if is_main_process():
                trainloader = tqdm(trainloader)
            for image, label in trainloader:
                model.train()
                image = image.cuda()
                label = label.cuda()

                # lr = lr_schedule(epoch + (i + 1) / len(trainloader))
                # opt.param_groups[0].update(lr=lr)
                lr = opt.state_dict()['param_groups'][0]['lr']
                # label_pe = torch.zeros([label.shape[0], 4], device=args.device)
                # for j in range(label.shape[0]):
                #     label_pe[j, int(label[j])] = 1
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    pred = model(image)
                    # loss = F.cosine_similarity(label_pe, pred.squeeze(), dim=1).mean()
                    loss_f = nn.CrossEntropyLoss()
                    loss = loss_f(pred, label)

                scaler.scale(loss).backward()
                if args.clip_norm:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(opt)
                scaler.update()

                train_loss += reduce_value(loss, average=True).item() * image.size(0)
                train_acc += (label == pred.argmax(dim=1)).sum()
                n += image.size(0)

            model.eval()
            test_acc, m = 0, 0
            with torch.no_grad():
                for image, label in testloader:
                    image = image.cuda()
                    label = label.cuda()
                    with torch.cuda.amp.autocast():
                        pred = model(image)
                    test_acc += (label == pred.argmax(dim=1)).sum()
                    m += image.size(0)

            scheduler.step()
            loss_list.append(train_loss / n)
            lr_list.append(lr)

            if is_main_process():
                print(
                    f'[Epoch: {epoch} | Train Loss: {train_loss / n:.4f},'
                    f'Train Acc: {train_acc / n:.4f},'
                    f'Test Acc: {test_acc / m:.4f},'
                    f'Time: {time.time() - start:.1f}, lr: {lr:.10f}')
                if (epoch + 1) % 50 == 0:
                    torch.save(model.module.state_dict(), os.path.join("shuffler_checkpoints", 'iter0', str(max_lr),
                                                                       str(epoch + 1) + "view_classifier.pth"))


if __name__ == '__main__':
    args = config_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.nprocs = len(args.gpus.split(','))
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
