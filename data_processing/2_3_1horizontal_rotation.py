import cv2
import numpy as np
from math import *
import pandas as pd
import random as rdm

def rotation(img,x1,y1,x2,y2,x3,y3,x4,y4,degree):
    # 逆时针旋转
    height, width = img.shape[:2]
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    #旋转中心
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    #平移
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    X1 = x1*matRotation[0][0]+y1*matRotation[0][1]+matRotation[0][2]
    Y1 = x1*matRotation[1][0]+y1*matRotation[1][1]+matRotation[1][2]
    X2 = x2*matRotation[0][0]+y2*matRotation[0][1]+matRotation[0][2]
    Y2 = x2*matRotation[1][0]+y2*matRotation[1][1]+matRotation[1][2]
    X3 = x3*matRotation[0][0]+y3*matRotation[0][1]+matRotation[0][2]
    Y3 = x3*matRotation[1][0]+y3*matRotation[1][1]+matRotation[1][2]
    X4 = x4*matRotation[0][0]+y4*matRotation[0][1]+matRotation[0][2]
    Y4 = x4*matRotation[1][0]+y4*matRotation[1][1]+matRotation[1][2]

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation,int(X1),int(Y1),int(X2),int(Y2),int(X3),int(Y3),int(X4),int(Y4)

def condition(condition,iftrue,iffalse):
    if condition:
        return iftrue
    else:
        return iffalse

all_use=np.load(r'.\data\correct_all_use.npy').tolist()

# 为all_use添加水平和竖直的标识，水平1，竖直2
for i in range(np.shape(all_use)[0]):
    x_len = np.mean([abs(int(all_use[i][2]) - int(all_use[i][0])), abs(int(all_use[i][4]) - int(all_use[i][6]))])
    y_len = np.mean([abs(int(all_use[i][5]) - int(all_use[i][3])), abs(int(all_use[i][7]) - int(all_use[i][1]))])
    all_use[i].append(condition(x_len>y_len,1,2))

all_use=np.array(all_use)

# 把水平的拎出来
horizontal_use=all_use[np.where(all_use[:,-1]=='1')]

# 纯水平组
horizontal_use_group=[]
for i in range(np.shape(horizontal_use)[0]):
    l=[horizontal_use[i][9],horizontal_use[i][11],horizontal_use[i][13],horizontal_use[i][14]]
    if l not in horizontal_use_group:
        horizontal_use_group.append(l)

#抽样旋转
# 读all_frame，存frame_rotation
rotation_detail=[]
for i in range(np.shape(horizontal_use_group)[0]):
    #待旋转集合, 一个集装箱序列
    one_un_rotation_jpg=list(set(all_use[np.where(np.logical_and(np.logical_and(all_use[:,9]==horizontal_use_group[i][0],all_use[:,11]==horizontal_use_group[i][1],np.array([int(a) for a in all_use[:,8]])<=9),\
                            all_use[:,13]==horizontal_use_group[i][2],all_use[:,14]==horizontal_use_group[i][3])),10].tolist()[0]))
    one_un_rotation=all_use[np.where(np.logical_and(np.logical_and(all_use[:,9]==horizontal_use_group[i][0],all_use[:,11]==horizontal_use_group[i][1],np.array([int(a) for a in all_use[:,8]])<=9),\
                            all_use[:,13]==horizontal_use_group[i][2],all_use[:,14]==horizontal_use_group[i][3]))]

    #把角度好的挑其中的1/3转
    if len(one_un_rotation_jpg)>=4:
        rotation_sample_jpg=np.array(rdm.sample(one_un_rotation_jpg,min(16,len(one_un_rotation_jpg)/3)))
        rotation_sample=[]
        for kk in range(np.shape(one_un_rotation)[0]):
            if one_un_rotation[kk,10] in rotation_sample_jpg:
                rotation_sample.append(one_un_rotation[kk].tolist())
        rotation_sample=np.array(rotation_sample)

        for sn in range(np.shape(rotation_sample_jpg)[0]):#一个帧
            image_path = r'E:\all_frame\%s' % rotation_sample_jpg[sn] #原始文件的帧存放路径
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            this_frame=rotation_sample[np.where(rotation_sample[:,10]==rotation_sample_jpg[sn])]
            degree_r=rdm.choice([-15,-14,-13,-12,-11,-10,-9,9,10,11,12,13,14,15])
            x_list = []
            y_list = []

            for line in range(np.shape(this_frame)[0]): #一个框
                '''
                '''
                img_rotation,X1,Y1,X2,Y2,X3,Y3,X4,Y4=rotation(image,int(this_frame[line,0]),int(this_frame[line,1]),int(this_frame[line,2]),int(this_frame[line,3]), \
                                                              int(this_frame[line, 4]),int(this_frame[line,5]),int(this_frame[line,6]),int(this_frame[line,7]),degree=degree_r)
                angle = degrees(atan(abs(float(Y1) - float(Y2)) / abs(float(X1) - float(X2))))
                rotation_detail.append(
                    [X1, Y1, X2, Y2, X3, Y3, X4, Y4, angle, this_frame[line, 9], this_frame[line, 10],
                     this_frame[line, 11], this_frame[line, 12], this_frame[line, 13], this_frame[line, 14]])
                x_list.append(X1)
                x_list.append(X2)
                x_list.append(X3)
                x_list.append(X4)
                y_list.append(Y1)
                y_list.append(Y2)
                y_list.append(Y3)
                y_list.append(Y4)

            '''
            cv2.imshow("rotation", img_rotation)
            cv2.waitKey()
            cv2.destroyAllWindows()
            '''
            cv2.imwrite(r'E:\frame_rotation\%s'%(this_frame[line,10]),img_rotation) #旋转后的图片存放路径
            '''
            min_x = max(min(x_list) - 10, 0)
            min_y = max(min(y_list) - 10, 0)
            max_x = min(max(x_list) + 10, img_rotation.shape[1])
            max_y = min(max(y_list) + 10, img_rotation.shape[0])
    
            cut = img_rotation[min_y:max_y, min_x:max_x]
    
            cv2.imshow("Original", cut)
            cv2.waitKey()
            cv2.destroyAllWindows()
            '''
        else:
            continue

np.save(r'.\data\rotaion_only',rotation_detail)

###########################

rotation_detail=np.load(r'.\data\rotaion_only.npy')
all_use=np.load(r'.\data\correct_all_use.npy').tolist() #改为新的

# 为all_use添加水平和竖直的标识，水平1，竖直2
for i in range(np.shape(all_use)[0]):
    x_len = np.mean([abs(int(all_use[i][2]) - int(all_use[i][0])), abs(int(all_use[i][4]) - int(all_use[i][6]))])
    y_len = np.mean([abs(int(all_use[i][5]) - int(all_use[i][3])), abs(int(all_use[i][7]) - int(all_use[i][1]))])
    all_use[i].append(condition(x_len>y_len,1,2))
all_use=np.array(all_use)


all_use_rotation=[]
for i in range(np.shape(all_use)[0]):
    if all_use[i,10] not in rotation_detail[:,10]:
        all_use_rotation.append(all_use[i].tolist())
all_use_rotation=np.array(all_use_rotation)
all_use_rotation=np.vstack((all_use_rotation,rotation_detail))

np.save(r'.\data\all_use_rotation',all_use_rotation)



