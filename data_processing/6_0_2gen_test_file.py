# -*- coding: utf-8 -*-
import numpy as np
import re
import sys
import pandas as pd
from math import *
import cv2
import os
import random as rdm
reload(sys)
sys.setdefaultencoding('utf-8')

'''
生成用于测试数据集样本和caffe文件
'''

def logic_and(list):
    result=list[0]
    for i in range(1,np.shape(list)[0]):
        result=np.logical_and(list[i],result)
    return result

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    if xB>xA and yB>yA:
        interArea = (abs(xB - xA) +1) * (abs(yB - yA) +1) #相交面积
    else:
        interArea=0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (abs(boxA[2] - boxA[0])+1 ) * (abs(boxA[3] - boxA[1])+1 ) # A面积
    boxBArea = (abs(boxB[2] - boxB[0])+1) * (abs(boxB[3] - boxB[1]) +1) # B面积

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return round(iou,2)

def rotation(img,degree):
    height, width = img.shape[:2]
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    #旋转中心
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation

def condition(condition,iftrue,iffalse):
    if condition:
        return iftrue
    else:
        return iffalse

def random_sample_angle(all_use):
    # 单纯的分组key键组合
    raw_group=[]
    for i in range(np.shape(all_use)[0]):
        x_len=np.mean([abs(int(all_use[i,2])-int(all_use[i,0])),abs(int(all_use[i,4])-int(all_use[i,6]))])
        y_len=np.mean([abs(int(all_use[i,5])-int(all_use[i,3])),abs(int(all_use[i,7])-int(all_use[i,1]))])
        l=[all_use[i,9],all_use[i,11],all_use[i,13]]
        if l not in raw_group:
            raw_group.append(l)

    #图片整个的倾斜角度看为是其中所有文本框对应角度的平均
    all_usepd=pd.DataFrame(all_use)
    all_usepd.columns = ['x1','y1','x2','y2','x3','y3','x4','y4','angle', 'id', 'pic','mp', 'frame','min_frame']
    pic_angel=all_usepd['angle'].astype(float).astype(int).groupby(all_usepd['pic']).mean()

    #分层采样，最大采样6组


    choose_file_all=[]
    for i in range(np.shape(raw_group)[0]):
        file_group_list=all_use[np.where(np.logical_and(np.logical_and(all_use[:,9]==raw_group[i][0],all_use[:,11]==raw_group[i][1]),\
                                                        all_use[:,13]==raw_group[i][2])),9:][0]
        file_group_list_uq=np.array(list(set([tuple(t) for t in file_group_list])))

        # 在group内部，根据角度是否大于9
        good_angle_file = []
        bad_angle_file = []
        for j in range(np.shape(file_group_list_uq)[0]):
            if pic_angel.loc['%s'%file_group_list_uq[j,1]]<9:
                good_angle_file.append(file_group_list_uq[j,1])
            else:
                bad_angle_file.append(file_group_list_uq[j,1])

        choose_list_file = []
        # 分类讨论如何随机采样
        good_num=np.shape(good_angle_file)[0]
        bad_num=np.shape(bad_angle_file)[0]

        if good_num+bad_num==8: #总数为8则全部采样
            choose_list_file = good_angle_file+bad_angle_file
            rdm.shuffle(choose_list_file)
        elif good_num<=4 and bad_num>=4 and good_num+bad_num>8: #好样本数小于4，全部采样好样本
            choose_list_file=good_angle_file+rdm.sample(bad_angle_file,8-good_num)
            rdm.shuffle(choose_list_file)
        elif good_num >= 4 and bad_num <= 4 and good_num+bad_num>8:#坏样本数小于4，全部采样坏样本
            choose_list_file = bad_angle_file + rdm.sample(good_angle_file, 8 - bad_num)
            rdm.shuffle(choose_list_file)
        elif good_num > 4 and bad_num > 4: #1次
            choose_list_file = rdm.sample(good_angle_file, 4) + rdm.sample(bad_angle_file, 4)
            rdm.shuffle(choose_list_file)
        else:
            pass

        if len(choose_list_file)>0:
            choose_file_all.append(choose_list_file)
        else:
            pass
    return choose_file_all

def sample_detail(choose_file_all,all_use):
    # choose_file_all 为选择的图片名称
    # all_use 为全部详细信息
    all_frame_pic=[]
    for one_group in choose_file_all:
        if len(np.shape(one_group))==1:# 组内只有一组样本采样
            frame_pic=[]
            for one_pic in one_group:
                frame_pic.append(all_use[np.where(all_use[:,10]==one_pic)].tolist())
            all_frame_pic.append(frame_pic)
        else: #组内有多组样本采样
            frame_sample=[]
            for one_sample_group in one_group:
                frame_pic = []
                for one_pic in one_sample_group:
                    frame_pic.append(all_use[np.where(all_use[:, 10] == one_pic)].tolist())
                frame_sample.append(frame_pic)
            all_frame_pic.append(frame_sample)
    return all_frame_pic


stat=np.load(r'.\data\6_0_1stat.npy')

#计算每组有多少个帧，只保留大于等于8张帧的
all_frame_pd=pd.DataFrame(stat)
all_frame_pd.columns= ['x1', 'y1', 'x2', 'y2', 'x3','y3','x4', 'y4', 'angle', 'id', 'pic','mp','frame','min_frame','hori1verti2']
all_group_num=all_frame_pd['pic'].groupby([all_frame_pd['id'],all_frame_pd['mp'],all_frame_pd['min_frame']]).size()

all_use_frame=[]
for i in range(np.shape(stat)[0]):
    if all_group_num.loc[stat[i,9]][stat[i,11]][stat[i,13]]>=8:
        all_use_frame.append(stat[i].tolist())
all_use_frame=np.array(all_use_frame)

# 取样8个，此处随机，不要轻易重跑

choose_file=random_sample_angle(all_use_frame)
choose_file_detail=sample_detail(choose_file,all_use_frame)

'''
此处随机，不要轻易save
np.save(r'.\data\choose_file_detail_test',choose_file_detail)
np.save(r'.\data\all_use_frame_test',all_use_frame)
'''

##########################为每一个测试序列添加4帧有部分缺失的
def logic_and(list):
    result=list[0]
    for i in range(1,np.shape(list)[0]):
        result=np.logical_and(list[i],result)
    return result

#test_group_structure=np.load(r'.\data\6_0_1test_group_structure.npy')
choose_file_detail=np.load(r'.\data\choose_file_detail_test.npy').tolist()  #[序列index][0:8选帧index][帧内标定index][]
all_use_frame=np.load(r'.\data\all_use_frame_test.npy')

for i in range(np.shape(choose_file_detail)[0]):
    group_index=[choose_file_detail[i][0][0][9],choose_file_detail[i][0][0][11],choose_file_detail[i][0][0][13]]
    this_group_frmae=all_use_frame[np.where(logic_and([all_use_frame[:,9]==group_index[0],all_use_frame[:,11]==group_index[1],all_use_frame[:,13]==group_index[2]]))]
    choose_pic=rdm.sample(list(set(this_group_frmae[:,10])),4) #随机选4个作为不全的
    rdm.shuffle(choose_pic)
    for k in range(4):
        choose_file_detail[i].append(this_group_frmae[np.where(this_group_frmae[:,10]==choose_pic[k])].tolist())

'''
此处随机，不要轻易save
np.save(r'.\data\choose_file_detail_test_add_incomplete',choose_file_detail)
'''

#############################切割箱号部分

def cut_image(image,n,min_x,min_y,max_x,max_y):
    cut = image[min_y:max_y, min_x:max_x]

    height, width = cut.shape[:2]
    if float(height)/float(width)>1.2: #改这里
        cut_rotation=rotation(cut,degree=90)
        cut_res=cv2.resize(cut_rotation,(192,64))
    else:
        cut_res = cv2.resize(cut, (192, 64))
    '''
    cv2.startWindowThread()
    cv2.imshow('image',cut)
    cv2.imshow('image1',cut4)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    for i in range(4):
        cv2.waitKey(1)
    '''

    path=r'..\test_cut_add_incomplete\%s\%s\%s'%(this_pic[0][9],this_pic[0][11],this_pic[0][13])
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass
    if n<=7:
        caffe_test_list = r'/data/liuxu/container_qan/test_data/%s/%s/%s/%s' % (this_pic[0][9], this_pic[0][11], this_pic[0][13],this_pic[0][10])
        #cv2.imwrite(r'..\test_cut_add_incomplete\%s\%s\%s\%s'%(this_pic[0][9],this_pic[0][11],this_pic[0][13],this_pic[0][10]),cut_res) #箱号、视频号、最小帧号、图
    else:
        caffe_test_list = r'/data/liuxu/container_qan/test_data/%s/%s/%s/%s' % (this_pic[0][9], this_pic[0][11], this_pic[0][13], 'add' + this_pic[0][10])
        #cv2.imwrite(r'..\test_cut_add_incomplete\%s\%s\%s\%s'%(this_pic[0][9],this_pic[0][11],this_pic[0][13],'add'+this_pic[0][10]),cut_res) #箱号、视频号、最小帧号、图

    return caffe_test_list

choose_file_detail=np.load(r'.\data\choose_file_detail_test_add_incomplete.npy')

caffe_test_list=[]
for nn in range(np.shape(choose_file_detail)[0]):
    for n in range(np.shape(choose_file_detail[nn])[0]):
        this_pic=choose_file_detail[nn][n]
        image = cv2.imread(r'..\all_frame_test\%s' % this_pic[0][10], cv2.IMREAD_COLOR)
        # [i for (i,j) in enumerate(all_pic) if j == 'EmbeddedNetDVR_172.26.4.207_8_20180525134735_20180525134832_1527230811919.mp4_812.jpg']
        x_list = []
        y_list = []
        for i in range(np.shape(this_pic)[0]):
            x_list.append(int(this_pic[i][0]))
            x_list.append(int(this_pic[i][2]))
            x_list.append(int(this_pic[i][4]))
            x_list.append(int(this_pic[i][6]))
            y_list.append(int(this_pic[i][1]))
            y_list.append(int(this_pic[i][3]))
            y_list.append(int(this_pic[i][5]))
            y_list.append(int(this_pic[i][7]))
        min_x = max(min(x_list) - 10, 0)
        min_y = max(min(y_list) - 10, 0)
        max_x = min(max(x_list) + 10, image.shape[1])
        max_y = min(max(y_list) + 10, image.shape[0])
        w=max_x-min_x
        h=max_y-min_y
        if n==8:
            min_y = min_y+(h / 6 +10)
        elif n==9:
            max_y =max_y- (h / 6 +10)
        elif n==10:
            min_x = min_x+(w / 6 +10)
        elif n==11:
            max_x = max_x-(w / 6+10)

        caffe_test_one=cut_image(image,n,min_x,min_y,max_x,max_y)
        caffe_test_list.append(caffe_test_one)

f = open(r'.\data\caffe_test_add_incomplete.txt','w')
for one in caffe_test_list:
    f.write('%s 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 38\n'%one)
f.close()



######################################### 补充一些不是箱号的文本图片
def loadDataSet(filename):   #读取文件，txt文档
    fr = open(filename,'r')
    numFeat = len(fr.readline().split(' ')) #自动检测特征的数目
    dataMat = []
    fr = open(filename,'r')
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split(' ')
        for i in range(numFeat):
            lineArr.append(curLine[i])
        dataMat.append(lineArr)
    return np.array(dataMat)

filename=r'.\data\test_add_no_number.txt'
no_label_test=loadDataSet(filename)

for i in range(np.shape(no_label_test)[0]):
    image = cv2.imread(r'E:\all_frame_test\%s.jpg' % no_label_test[i,0], cv2.IMREAD_COLOR)
    cut = image[int(no_label_test[i,2])-10:int(no_label_test[i,4])+10, int(no_label_test[i,1])-10:int(no_label_test[i,3])+10]
    height, width = cut.shape[:2]
    if height/width>1.1: #改这里
        cut_rotation=rotation(cut,degree=90)
        cut_res=cv2.resize(cut_rotation,(192,64))
    else:
        cut_res = cv2.resize(cut, (192, 64))

    '''
    cv2.imshow("cut", cut_res)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    path=r'E:\test_cut\OTHER\1'
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass
    #cv2.imwrite(r'E:\test_cut\OTHER\0\1\add_%s.jpg'%(no_label_test[i,0]),cut_res)
    caffe_test_list.append([r'/data/liuxu/container_qan/test_data/other/0/1/add_%s.jpg' % (no_label_test[i,0])])


# caffe 文档
f=open(r'.\data\caffe_test.txt','w')
for line in caffe_test_list:
    f.write('%s 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 38\n' % line)
f.close()

