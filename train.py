import os
import math
import argparse
import random
import logging
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from utils import extra_util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
from torchscan import summary
from fvcore.nn import FlopCountAnalysis

def valid(val_loader, model, opt, logger):
    with torch.no_grad():
        network = model.netG
        flops = FlopCountAnalysis(network, torch.zeros(1, 3, 3840, 2160).cuda())
        print(flops.total())
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(num_params)

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_deltaITP = 0.0
    idx = 0
    for val_data in val_loader:
        idx += 1
        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
        img_dir = opt['path']['val_images'] # img_dir = os.path.join(opt['path']['val_images'], img_name)
        util.mkdir(img_dir)

        # model.netG.recon_trunk_lpf[0].img_name = img_name
        model.feed_data(val_data)
        model.test()

        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals['SR'], out_type=np.uint16)  # uint16
        gt_img = util.tensor2img(visuals['GT'], out_type=np.uint16)  # uint16

        # Save SR images for reference
        # if opt['datasets']['val']['save_img']:
        if opt['path']['pretrain_model_G'] is not None:
            save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name)) # save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
            util.save_img(sr_img, save_img_path)

        hq_ndy_bgr = (sr_img / 65535).astype(np.float32)
        gt_ndy_bgr = (gt_img / 65535).astype(np.float32)

        hq_ndy_rgb = hq_ndy_bgr[...,::-1]
        gt_ndy_rgb = gt_ndy_bgr[...,::-1]

        psnr = extra_util.calculate_psnr(img1=hq_ndy_bgr, img2=gt_ndy_bgr)
        ssim = extra_util.calculate_ssim(img=hq_ndy_bgr * 255, img2=gt_ndy_bgr * 255)
        deltaITP = extra_util.calculate_hdr_deltaITP(img1=hq_ndy_bgr, img2=gt_ndy_bgr)
        avg_psnr += psnr
        avg_ssim += ssim
        avg_deltaITP += deltaITP

        logger.info('# Index: {:03d} # PSNR: {:.4f} # SSIM: {:.4f} # deltaITP: {:.4f}'.format(idx, psnr, ssim, deltaITP))
        
        # torch.cuda.empty_cache()

        # # calculate PSNR
        # gt_img = gt_img / 65535.
        # sr_img = sr_img / 65535.
        # avg_psnr += util.calculate_psnr(sr_img, gt_img)

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_deltaITP = avg_deltaITP / idx

    return avg_psnr, avg_ssim, avg_deltaITP

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--debug', type=bool, default=False, help='whether to perform debug mode for VSCode')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    if parser.parse_args().debug == True:
        opt['gpu_ids'] = [0]
        opt['dist'] = False
        opt['datasets']['train']['n_workers'] = 0
        opt['datasets']['train']['batch_size'] = 2

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir=os.path.join(opt['path']['root'], 'tb_logger', opt['name']))
            #tb_logger = SummaryWriter(log_dir=  + '/tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    if_break = False
    if current_step < total_iters:
        #### training
        logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
        first_time = True
        for epoch in range(start_epoch, total_epochs + 1):
            if opt['dist']:
                train_sampler.set_epoch(epoch)
            for _, train_data in enumerate(train_loader):
                if first_time:
                    start_time = time.time()
                    first_time = False
                current_step += 1
                if current_step > total_iters:
                    if_break = True
                    break
                
                #### training
                model.feed_data(train_data)
                model.optimize_parameters(current_step)
                
                #### update learning rate
                model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

                #### log
                if current_step % opt['logger']['print_freq'] == 0:
                    end_time = time.time()
                    logs = model.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, time:{:.3f}> '.format(
                        epoch, current_step, model.get_current_learning_rate(), end_time-start_time)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        # tensorboard logger
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        logger.info(message)
                    start_time = time.time()

                #### save models and training states
                if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                    if rank <= 0:
                        logger.info('Saving models and training states.')
                        model.save(current_step)
                        model.save_training_state(epoch, current_step)

                # validation
                if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                    avg_psnr, avg_ssim, avg_deltaITP = valid(val_loader, model, opt, logger)
                    
                    # log
                    logger.info('# Validation # PSNR: {:.4f} # SSIM: {:.4f} # deltaITP: {:.4f}'.format(avg_psnr, avg_ssim, avg_deltaITP))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> PSNR: {:.4f} # SSIM: {:.4f} # deltaITP: {:.4f}'.format(
                        epoch, current_step, avg_psnr, avg_ssim, avg_deltaITP))
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)

            if if_break == True:
                break

        if rank <= 0:
            model_parameters = filter(lambda p: p.requires_grad, model.netG.parameters())
            params = int(sum([np.prod(p.size()) for p in model_parameters]))
            logger.info('Params: {:3.4f} [M]'.format((params / 1024**2)))
            logger.info('Saving the final model.')
            model.save('latest')
            logger.info('End of training.')
    else:
        avg_psnr, avg_ssim, avg_deltaITP = valid(val_loader, model, opt, logger)
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> PSNR: {:.4f} # SSIM: {:.4f} # deltaITP: {:.4f}'.format(start_epoch, current_step, avg_psnr, avg_ssim, avg_deltaITP))
    
    exit()

if __name__ == '__main__':
    main()