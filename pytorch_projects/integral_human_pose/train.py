import os
import pprint
import copy
import time
import matplotlib
from torch.utils.data import DataLoader

matplotlib.use('Agg')

# define project dependency
import _init_paths

# common
from common.speedometer import Speedometer
from common.utility.logger import create_logger
from common.utility.visualization import plot_LearningCurve

# project dependence
from core.loader import hm36_Dataset, mpii_hm36_Dataset
from common_pytorch.dataset.all_dataset import *
from common_pytorch.config_pytorch import update_config_from_file, update_config_from_args, s_args, s_config \
    , s_config_file
from common_pytorch.optimizer import get_optimizer
from common_pytorch.io_pytorch import save_model, save_lowest_vloss_model
from common_pytorch.common_loss.balanced_parallel import DataParallelModel, DataParallelCriterion
from common_pytorch.net_modules import trainNet, validNet, evalNet

# import dynamic config
exec('from blocks.' + s_config.pytorch.block + \
     ' import get_default_network_config, get_pose_net, init_pose_net')
exec('from loss.' + s_config.pytorch.loss + \
     ' import get_default_loss_config, get_loss_func, get_label_func, get_result_func, get_merge_func')


def main():
    # parsing specific config
    config = copy.deepcopy(s_config)
    config.network = get_default_network_config()
    config.loss = get_default_loss_config()

    config = update_config_from_file(config, s_config_file, check_necessity=True)
    config = update_config_from_args(config, s_args)

    # create log and path
    final_output_path, final_log_path, logger = create_logger(s_config_file, config.dataset.train_image_set,
                                                              config.pytorch.output_path, config.pytorch.log_path)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

  # define devices create multi-GPU context
    os.environ["CUDA_VISIBLE_DEVICES"] = config.pytorch.gpus  # a safer method
    devices = [int(i) for i in config.pytorch.gpus.split(',')]
    logger.info("Using Devices: {}".format(str(devices)))

    # lable, loss, metric and result
    logger.info("Defining lable, loss, metric and result")
    label_func = get_label_func(config.loss)
    loss_func = get_loss_func(config.loss)
    loss_func = DataParallelCriterion(loss_func)
    result_func = get_result_func(config.loss)
    merge_flip_func = get_merge_func(config.loss)

    # dataset, basic imdb
    logger.info("Creating dataset")
    train_imdbs = []
    valid_imdbs = []
    for n_db in range(0, len(config.dataset.name)):
        train_imdbs.append(
            eval(config.dataset.name[n_db])(config.dataset.train_image_set[n_db], config.dataset.path[n_db],
                                            config.train.patch_width, config.train.patch_height,
                                            config.train.rect_3d_width, config.train.rect_3d_height))
        valid_imdbs.append(
            eval(config.dataset.name[n_db])(config.dataset.test_image_set[n_db], config.dataset.path[n_db],
                                            config.train.patch_width, config.train.patch_height,
                                            config.train.rect_3d_width, config.train.rect_3d_height))

    batch_size = len(devices) * config.dataiter.batch_images_per_ctx

    # basic data_loader unit
    dataset_name = ""
    for n_db in range(0, len(config.dataset.name)):
        dataset_name = dataset_name + config.dataset.name[n_db] + "_"
    dataset_train = \
        eval(dataset_name + "Dataset")(train_imdbs, True, '', config.train.patch_width, config.train.patch_height,
                                       config.train.rect_3d_width, config.train.rect_3d_height, batch_size,
                                       config.dataiter.mean, config.dataiter.std, config.aug, label_func, config.loss)

    dataset_valid = \
        eval(config.dataset.name[config.dataiter.target_id] + "_Dataset")([valid_imdbs[config.dataiter.target_id]],
                                                                          False, config.train.patch_width,
                                                                          config.train.patch_height,
                                                                          config.train.rect_3d_width,
                                                                          config.train.rect_3d_height, batch_size,
                                                                          config.dataiter.mean, config.dataiter.std,
                                                                          config.aug, label_func, config.loss)

    train_data_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True,
                                   num_workers=config.dataiter.threads, drop_last=True)
    valid_data_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False,
                                   num_workers=config.dataiter.threads, drop_last=False)

    # prepare network
    logger.info("Creating network")
    joint_num = dataset_train.joint_num
    assert dataset_train.joint_num == dataset_valid.joint_num
    net = get_pose_net(config.network, joint_num)
    init_pose_net(net, config.network)
    net = DataParallelModel(net).cuda()
    model_prefix = os.path.join(final_output_path, config.train.model_prefix)
    logger.info("Net total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))

    # Optimizer
    logger.info("Creating optimizer")
    optimizer, scheduler = get_optimizer(config.optimizer, net)

    # train and valid
    vloss_min = 10000000.0
    train_loss = []
    valid_loss = []
    logger.info("Train DB size: {}; Valid DB size: {}.".format(int(len(dataset_train)), int(len(dataset_valid))))
    for epoch in range(config.train.begin_epoch, config.train.end_epoch + 1):
        scheduler.step()
        logger.info(
            "Working on {}/{} epoch || LearningRate:{} ".format(epoch, config.train.end_epoch, scheduler.get_lr()[0]))
        speedometer = Speedometer(train_data_loader.batch_size, config.pytorch.frequent, auto_reset=False)

        beginT = time.time()
        tloss = trainNet(epoch, train_data_loader, net, optimizer, config.loss, loss_func, speedometer)
        endt1 = time.time() - beginT

        beginT = time.time()
        preds_in_patch_with_score, vloss = \
            validNet(valid_data_loader, net, config.loss, result_func, loss_func, merge_flip_func,
                     config.train.patch_width, config.train.patch_height, devices,
                     valid_imdbs[config.dataiter.target_id].flip_pairs, flip_test=False)
        endt2 = time.time() - beginT

        beginT = time.time()
        evalNet(epoch, preds_in_patch_with_score, valid_data_loader, valid_imdbs[config.dataiter.target_id],
                config.train.patch_width, config.train.patch_height, config.train.rect_3d_width,
                config.train.rect_3d_height, final_output_path)
        endt3 = time.time() - beginT
        logger.info('One epoch training %.1fs, validation %.1fs, evaluation %.1fs ' % (endt1, endt2, endt3))

        train_loss.append(tloss)
        valid_loss.append(vloss)

        if vloss < vloss_min:
            vloss_min = vloss
            save_lowest_vloss_model({
                'epoch': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss
            }, model_prefix, logger)

        if epoch % (config.train.end_epoch // 10) == 0 \
                or epoch == config.train.begin_epoch \
                or epoch == config.train.end_epoch:
            save_model({
                'epoch': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss
            }, model_prefix, logger, epoch)

        # jobName = os.path.basename(s_args.cfg).split('.')[0]
        # plot_LearningCurve(train_loss, valid_loss, config.pytorch.log_path, jobName)
        # plot_LearningCurve(train_loss, valid_loss, final_log_path, "Learning Curve")


if __name__ == "__main__":
    main()
