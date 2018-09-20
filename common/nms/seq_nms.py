# -*- coding: utf-8 -*-
import numpy as np
from nms import nms
import time
import copy

CONF_THRESH = 0.5

'''
修改检测结果格式，用作后续处理
第一维：种类
第二维：帧
第三维：bbox
第四维：x1,y1,x2,y2,score
'''

def createLinks(dets_all, thresh = 0.3):
    links_all = []

    frame_num = len(dets_all[0])
    cls_num = len(dets_all)

    for cls_ind in range(cls_num):
        links_cls = []
        for frame_ind in range(frame_num-1):
            dets1 = dets_all[cls_ind][frame_ind]
            dets2 = dets_all[cls_ind][frame_ind+1]
            box1_num = len(dets1)
            box2_num = len(dets2)

            if frame_ind == 0:
                areas1 = np.empty(box1_num)
                for box1_ind, box1 in enumerate(dets1):
                    areas1[box1_ind] = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
            else:
                areas1 = areas2
            areas2 = np.empty(box2_num)
            for box2_ind, box2 in enumerate(dets2):
                areas2[box2_ind] = (box2[2]-box2[0]+1) * (box2[3]-box2[1]+1)

            links_frame = []
            for box1_ind, box1 in enumerate(dets1):
                box1_area = areas1[box1_ind]
                x1 = np.maximum(box1[0], dets2[:,0])
                y1 = np.maximum(box1[1], dets2[:,1])
                x2 = np.minimum(box1[2], dets2[:,2])
                y2 = np.minimum(box1[3], dets2[:,3])
                w = np.maximum(0.0, x2 - x1 + 1)
                h = np.maximum(0.0, y2 - y1 + 1)
                inter = w * h
                ovrs = inter / (box1_area + areas2 - inter)
                # the list contains the index of boxes in the next frame that link to it
                links_box = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if ovr >= thresh]  # thresh ???
                links_frame.append(links_box)
            links_cls.append(links_frame)
        links_all.append(links_cls)
    return links_all


def maxPath(dets_all, links_all, thresh):
    for cls_ind, links_cls in enumerate(links_all):
        dets_cls = dets_all[cls_ind]
        while True:
            rootindex, maxpath, maxsum = findMaxPath(links_cls, dets_cls)
            if len(maxpath) <= 1:
                break
            rescore(dets_cls, rootindex, maxpath, maxsum)
            deleteLink(dets_cls, links_cls, rootindex, maxpath, thresh)


def seq_nms(dets_all, thresh):
    # nms = py_nms_wrapper(thresh)
    links = createLinks(dets_all, thresh)
    maxPath(dets_all, links, thresh)
    # for cls_ind, dets_cls in enumerate(dets_all):
    #     for frame_ind, dets in enumerate(dets_cls):
    #         keep = nms(dets, thresh)
    #         dets_all[cls_ind][frame_ind] = dets[keep, :]


def findMaxPath(links, dets):
    maxpaths = []   #保存从每个结点到最后的最大路径与分数
    roots = []  #保存所有的可作为独立路径进行最大路径比较的路径
    maxpaths.append([ (box[4], [ind]) for ind, box in enumerate(dets[-1])])
    for link_ind, link in enumerate(links[::-1]):   #每一帧与后一帧的link，为一个list
        curmaxpaths = []
        linkflags = np.zeros(len(maxpaths[0]), int)
        det_ind = len(links)-link_ind-1
        for ind, linkboxes in enumerate(link):  #每一帧中每个box的link，为一个list
            if linkboxes == []:
                curmaxpaths.append((dets[det_ind][ind][4], [ind]))
                continue
            linkflags[linkboxes] = 1
            prev_ind = np.argmax([maxpaths[0][linkbox][0] for linkbox in linkboxes])
            prev_score = maxpaths[0][linkboxes[prev_ind]][0]
            prev_path = copy.copy(maxpaths[0][linkboxes[prev_ind]][1])
            prev_path.insert(0, ind)
            curmaxpaths.append((dets[det_ind][ind][4]+prev_score, prev_path))
        root = [maxpaths[0][ind] for ind, flag in enumerate(linkflags) if flag == 0]
        roots.insert(0, root)
        maxpaths.insert(0, curmaxpaths)
    roots.insert(0, maxpaths[0])
    maxscore = 0
    maxpath = []
    rootindex = 0
    for index, paths in enumerate(roots):
        if paths == []:
            continue
        maxindex = np.argmax([path[0] for path in paths])
        if paths[maxindex][0] > maxscore:
            maxscore = paths[maxindex][0]
            maxpath = paths[maxindex][1]
            rootindex = index
    return rootindex, maxpath, maxscore


def rescore(dets, rootindex, maxpath, maxsum):
    newscore = maxsum/len(maxpath)
    for i, box_ind in enumerate(maxpath):
        dets[rootindex+i][box_ind][4] = newscore


def deleteLink(dets, links, rootindex, maxpath, thresh=0.3):
    for i, box_ind in enumerate(maxpath):
        areas = [(box[2]-box[0]+1)*(box[3]-box[1]+1) for box in dets[rootindex+i]]
        area1 = areas[box_ind]
        box1 = dets[rootindex+i][box_ind]
        x1 = np.maximum(box1[0], dets[rootindex+i][:, 0])
        y1 = np.maximum(box1[1], dets[rootindex+i][:, 1])
        x2 = np.minimum(box1[2], dets[rootindex+i][:, 2])
        y2 = np.minimum(box1[3], dets[rootindex+i][:, 3])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        inter = w * h
        ovrs = inter / (area1 + areas - inter)
        deletes = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if ovr >= thresh] #保存待删除的box的index
        if rootindex+i < len(links): #除了最后一帧，置box_ind的box的link为空
            for delete_ind in deletes:
                links[rootindex+i][delete_ind] = []
            # np.delete(dets[rootindex+i], deletes, 0)
        if i > 0 or rootindex > 0:
            for priorbox in links[rootindex+i-1]: #将前一帧指向box_ind的link删除
                for delete_ind in deletes:
                    if delete_ind in priorbox:
                        priorbox.remove(delete_ind)


