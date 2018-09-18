import os
import torch
import logging
import numpy as np
import pickle

from common.speedometer import BatchEndParam
from common.utility.image_processing_cv import trans_coords_from_patch_to_org_3d
from common.utility.image_processing_cv import rescale_pose_from_patch_to_camera

from torch.nn.parallel.scatter_gather import gather

from common_pytorch.common_loss.loss_recorder import LossRecorder
from common.utility.image_processing_cv import flip


def trainNet(nth_epoch, train_data_loader, network, optimizer, loss_config, loss_func, speedometer=None):
    """

    :param nth_epoch:
    :param train_data_loader: batch_size, dataset.db_length
    :param network:
    :param optimizer:
    :param loss_config:
    :param loss_func:
    :param speedometer:
    :param tensor_board:
    :return:
    """
    network.train()

    loss_recorder = LossRecorder()
    for idx, _data in enumerate(train_data_loader):
        batch_data = _data[0]
        batch_label = _data[1]
        batch_label_weight = _data[2]

        optimizer.zero_grad()

        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        batch_label_weight = batch_label_weight.cuda()

        preds = network(batch_data)
        del batch_data

        loss = loss_func(preds, batch_label, batch_label_weight)
        del batch_label, batch_label_weight
        del preds

        loss.backward()

        optimizer.step()

        loss_recorder.update(loss.detach(), train_data_loader.batch_size)
        del loss

        if speedometer != None:
            speedometer(BatchEndParam(epoch=nth_epoch, nbatch=idx,
                                      total_batch=train_data_loader.dataset.db_length // train_data_loader.batch_size,
                                      add_step=True, eval_metric=None, loss_metric=loss_recorder, locals=locals()))

    return loss_recorder.get_avg()


def validNet(valid_data_loader, network, loss_config, result_func, loss_func, merge_flip_func,
             patch_width, patch_height, devices, flip_pair, flip_test=True, flip_fea_merge=True):
    """

    :param nth_epoch:
    :param valid_data_loader:
    :param network:
    :param loss_config:
    :param result_func:
    :param loss_func:
    :param patch_size:
    :param devices:
    :param tensor_board:
    :return:
    """
    print('in valid')
    network.eval()

    loss_recorder = LossRecorder()

    preds_in_patch_with_score = []
    with torch.no_grad():
        for idx, _data in enumerate(valid_data_loader):
            batch_data = _data[0]

            if batch_data.shape[1] != 3:
                flip_test = False

            batch_label = _data[1]
            batch_label_weight = _data[2]

            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
            batch_label_weight = batch_label_weight.cuda()

            preds = network(batch_data)

            if flip_test:
                batch_data_flip = flip(batch_data, dims=3)
                preds_flip = network(batch_data_flip)

            del batch_data

            # loss
            loss = loss_func(preds, batch_label, batch_label_weight)
            del batch_label

            loss_recorder.update(loss.detach(), valid_data_loader.batch_size)
            del loss

            # get joint result in patch image
            if len(devices) > 1:
                preds = gather(preds, 0)

            if flip_test:
                if len(devices) > 1:
                    preds_flip = gather(preds_flip, 0)
                if flip_fea_merge:
                    preds = merge_flip_func(preds, preds_flip, flip_pair)
                    preds_in_patch_with_score.append(result_func(loss_config, patch_width, patch_height, preds))
                else:
                    pipws = result_func(loss_config, patch_width, patch_height, preds)
                    pipws_flip = result_func(loss_config, patch_width, patch_height, preds_flip)
                    pipws_flip[:, :, 0] = patch_width - pipws_flip[:, :, 0] - 1
                    for pair in flip_pair:
                        tmp = pipws_flip[:, pair[0], :].copy()
                        pipws_flip[:, pair[0], :] = pipws_flip[:, pair[1], :].copy()
                        pipws_flip[:, pair[1], :] = tmp.copy()
                    preds_in_patch_with_score.append((pipws + pipws_flip) * 0.5)

            else:
                preds_in_patch_with_score.append(result_func(loss_config, patch_width, patch_height, preds))
            del preds, batch_label_weight

    _p = np.asarray(preds_in_patch_with_score)
    _p = _p.reshape((_p.shape[0] * _p.shape[1], _p.shape[2], _p.shape[3]))
    preds_in_patch_with_score = _p[0: valid_data_loader.dataset.num_samples]

    return preds_in_patch_with_score, loss_recorder.get_avg()


def evalNet(nth_epoch, preds_in_patch_with_score, valid_data_loader, imdb, patch_width, patch_height
            , rect_3d_width, rect_3d_height, final_output_path):
    """
    :param nth_epoch:
    :param preds_in_patch_with_score:
    :param gts:
    :param convert_func:
    :param eval_func:
    :return:
    """

    print("in eval")
    # From patch to original image coordinate system
    imdb_list = valid_data_loader.dataset.db
    preds_in_img_with_score = []
    for n_sample in range(valid_data_loader.dataset.num_samples):
        preds_in_img_with_score.append(
            trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[n_sample], imdb_list[n_sample]['center_x'],
                                              imdb_list[n_sample]['center_y'], imdb_list[n_sample]['width'],
                                              imdb_list[n_sample]['height'], patch_width, patch_height,
                                              rect_3d_width, rect_3d_height))

    preds_in_img_with_score = np.asarray(preds_in_img_with_score)

    # Evaluate
    name_value = imdb.evaluate(preds_in_img_with_score.copy(), final_output_path)
    for name, value in name_value:
        logging.info('Epoch[%d] Validation-%s=%f', nth_epoch, name, value)


def evalNetChallenge(nth_epoch, preds_in_patch, valid_data_loader, imdb, final_output_path):
    """
    :param nth_epoch:
    :param preds_in_patch_with:
    :param gts:
    :param convert_func:
    :param eval_func:
    :return:
    """
    target_bone_length = 4502.881  # train+val
    # target_bone_length = 4522.828     # train
    # target_bone_length = 4465.869     # val
    print("in eval")

    # 4. From patch to original image coordinate system
    imdb_list = valid_data_loader.dataset.db
    preds_in_camera_space = []
    for n_sample in range(valid_data_loader.dataset.num_samples):
        preds_in_camera_space.append(
            rescale_pose_from_patch_to_camera(preds_in_patch[n_sample],
                                              target_bone_length,
                                              imdb_list[n_sample]['parent_ids']))

    preds_in_camera_space = np.asarray(preds_in_camera_space)[:, :, 0:3]

    # 5. Convert joint type
    preds_in_camera_space_cvt = preds_in_camera_space.copy()

    # 6. Evaluate
    name_value = imdb.evaluate(preds_in_camera_space_cvt, final_output_path)
    for name, value in name_value:
        logging.info('Epoch[%d] Validation-%s=%f', nth_epoch, name, value)

    return preds_in_patch
