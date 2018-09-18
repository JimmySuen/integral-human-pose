import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_LearningCurve(train_loss, valid_loss, log_path, jobName):
    '''
    Use matplotlib to plot learning curve at the end of training
    train_loss & valid_loss must be 'list' type
    '''
    plt.figure(figsize=(12,5))
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    epochs = np.arange(len(train_loss))
    plt.plot(epochs, np.array(train_loss), 'r', label='train')
    plt.plot(epochs, np.array(valid_loss), 'b', label='valid')
    plt.legend()  
    plt.grid()  
    plt.savefig(os.path.join(log_path, jobName + '.png'))

def plt_show_joints(img, pts, pts_vis=None, color='yo'):
    # imshow(img)
    plt.imshow(img)
    if pts_vis == None:
        pts_vis = np.ones(pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        if pts_vis[i, 0] > 0:
            plt.plot(pts[i, 0], pts[i, 1], color)
    # plt.axis('off')

def cv_draw_joints(im, kpt, vis, flip_pair_ids, color_left=(255, 0, 0), color_right=(0, 255, 0), radius=2):
    for ipt in range(0, kpt.shape[0]):
        if vis[ipt, 0]:
            cv2.circle(im, (int(kpt[ipt, 0] + 0.5), int(kpt[ipt, 1] + 0.5)), radius, color_left, -1)
    for i in range(0, flip_pair_ids.shape[0]):
        id = flip_pair_ids[i][0]
        if vis[id, 0]:
            cv2.circle(im, (int(kpt[id, 0] + 0.5), int(kpt[id, 1] + 0.5)), radius, color_right, -1)

def plot_3d_skeleton(ax, kpt_3d, kpt_3d_vis, parent_ids, flip_pair_ids, title, patch_width, patch_height, c0='r',c1='b',c2='g'):
    x_r = np.array([0, patch_width], dtype=np.float32)
    y_r = np.array([0, patch_height], dtype=np.float32)
    z_r = np.array([-patch_width / 2.0, patch_width / 2.0], dtype=np.float32)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    # joints
    X = kpt_3d[:, 0]
    Y = kpt_3d[:, 1]
    Z = kpt_3d[:, 2]
    vis_X = kpt_3d_vis[:, 0]
    #
    for i in range(0, kpt_3d.shape[0]):
        if vis_X[i]:
            ax.scatter(X[i], Z[i], -Y[i], c=c0, marker='o')
        x = np.array([X[i], X[parent_ids[i]]], dtype=np.float32)
        y = np.array([Y[i], Y[parent_ids[i]]], dtype=np.float32)
        z = np.array([Z[i], Z[parent_ids[i]]], dtype=np.float32)

        if vis_X[i] and vis_X[parent_ids[i]]:
            c = c1 # 'b'
            for j in range(0, flip_pair_ids.shape[0]):
                if i == flip_pair_ids[j][0]:
                    c = c2 # 'g'
                    break
            ax.plot(x, z, -y, c=c)
    ax.plot(x_r, z_r, -y_r, c='y')
    # ax.plot(np.array([np.min(X), np.max(X)]), np.array([np.min(Z), np.max(Z)]), -np.array([np.min(Y), np.max(Y)]), c='y')
    ax.set_title(title)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

def debug_vis(img_path, bbox=list(), pose=list()):
    cv_img_patch_show = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if len(bbox) > 0:
        c_x, c_y, width, height = bbox
        pt1 = (int(c_x - 1.0 * width / 2), int(c_y - 1.0 * height / 2))
        pt2 = (int(c_x + 1.0 * width / 2), int(c_y + 1.0 * height / 2))
        cv2.rectangle(cv_img_patch_show, pt1, pt2, (0, 128, 255), 3)

    #TODO: add flip pairs
    if len(pose) > 0:
        jts_3d, jts_3d_vis = pose
        for pt, pt_vis in zip(jts_3d, jts_3d_vis):
            if pt_vis[0] > 0:
                cv2.circle(cv_img_patch_show, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)

    cv2.imshow('debug visualization', cv_img_patch_show)
    cv2.waitKey(0)

def vis_compare_3d_pose(pose_a, pose_b):
    buff_large_1 = np.zeros((32, 3))
    buff_large_2 = np.zeros((32, 3))
    buff_large_1[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :] = pose_a[:-1]
    buff_large_2[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :] = pose_b[:-1]

    pose3D_1 = buff_large_1.transpose()
    pose3D_2 = buff_large_2.transpose()

    kin = np.array(
        [[0, 12], [12, 13], [13, 14], [15, 14], [13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27], [0, 1],
         [1, 2],
         [2, 3], [0, 6], [6, 7], [7, 8]])

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure(1, figsize=(10, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.view_init(azim=-90, elev=15)

    for link in kin:
        ax.plot(pose3D_1[0, link], pose3D_1[2, link], -pose3D_1[1, link],
                linestyle='--', marker='o', color='green', linewidth=3.0)
        ax.plot(pose3D_2[0, link], pose3D_2[2, link], -pose3D_2[1, link],
                linestyle='-', marker=',', color='red', linewidth=3.0)
    ax.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    X = pose3D_1[0, :]
    Y = pose3D_1[2, :]
    Z = -pose3D_1[1, :]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax_2d = fig.add_subplot(122)
    for link in kin:
        ax_2d.plot(pose3D_1[0, link], -pose3D_1[1, link],
                   linestyle='--', marker='o', color='green', linewidth=3.0)
        ax_2d.plot(pose3D_2[0, link], -pose3D_2[1, link],
                   linestyle='-', marker=',', color='red', linewidth=3.0)
    ax_2d.set_xlabel('X')
    ax_2d.set_ylabel('Y')
    plt.show()