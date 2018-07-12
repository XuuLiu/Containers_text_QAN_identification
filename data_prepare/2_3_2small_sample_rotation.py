import numpy as np
import pandas as pd
import random as rdm
import cv2
from math import *
import re

#帧数小于8的序列进行旋转填充到8

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


all_use_rotation=np.load(r'.\data\all_use_rotation.npy')

all_use_rotationpd=pd.DataFrame(np.load(r'.\data\all_use_rotation.npy'))
all_use_rotationpd.columns = ['x1','y1','x2','y2','x3','y3','x4','y4','angle', 'id', 'pic','mp', 'frame','min_frame','vertical2']
frame_count=all_use_rotationpd['pic'].drop_duplicates().groupby([all_use_rotationpd['id'],all_use_rotationpd['mp'],all_use_rotationpd['min_frame'],all_use_rotationpd['vertical2']]).size()

#找出总帧数小于8帧的所有
frame_less8=[]
for i in range(np.shape(all_use_rotation)[0]):
    if frame_count[all_use_rotation[i,9]][all_use_rotation[i,11]][all_use_rotation[i,13]][all_use_rotation[i,14]]<8 and \
            frame_count[all_use_rotation[i,9]][all_use_rotation[i,11]][all_use_rotation[i,13]][all_use_rotation[i,14]]>=4:
        frame_less8.append(all_use_rotation[i].tolist())
frame_less8=np.array(frame_less8)

# 单纯的可填补的组
frame_less8_group=[]
for i in range(np.shape(frame_less8)[0]):
    g=[frame_less8[i,9],frame_less8[i,11],frame_less8[i,13],frame_less8[i,14]]
    if g not in frame_less8_group:
        frame_less8_group.append(g)

frame_less8_group.sort(key=lambda l:(l[0],l[1],l[2],l[3]))
frame_less8_group=np.array(frame_less8_group)

# 存到frame_rotation
out_add_group=np.zeros([1,15])
for one_group in frame_less8_group:
    one_group_frame=frame_less8[np.where(np.logical_and(frame_less8[:,9]==one_group[0],frame_less8[:,11]==one_group[1],\
                                                              np.logical_and(frame_less8[:,13]==one_group[2],frame_less8[:,14]==one_group[3])))].tolist()
    one_group_frame.sort(key=lambda l:(l[9],l[11],l[12]))
    one_group_frame=np.array(one_group_frame)
    sample_num=np.shape(list(set(one_group_frame[:,10])))[0]
    create_sample=rdm.sample(list(set(one_group_frame[:,10])), 8 - sample_num)
    max_frame=max(one_group_frame[:,12])
    for sn in range(len(create_sample)):
        image = cv2.imread(r'E:\frame_rotation\%s' % create_sample[sn], cv2.IMREAD_COLOR)
        try:
            if image==None:
                image = cv2.imread(r'E:\all_frame\%s' % create_sample[sn], cv2.IMREAD_COLOR)
            else:
                pass
        except:
            pass

        this_frame = one_group_frame[np.where(one_group_frame[:, 10] == create_sample[sn])]
        degree_r = rdm.choice([-15,-14,-13,-12,-11,-10,-9,9,10,11,12,13,14,15])
        x_list = []
        y_list = []

        for line in range(np.shape(this_frame)[0]):  # 一个框
            '''
            cv2.imshow("Original", image)
            cv2.waitKey()
            cv2.destroyAllWindows()            
            '''
            img_rotation, X1, Y1, X2, Y2, X3, Y3, X4, Y4 = rotation(image, int(this_frame[line, 0]),
                                                                    int(this_frame[line, 1]), int(this_frame[line, 2]),
                                                                    int(this_frame[line, 3]), \
                                                                    int(this_frame[line, 4]), int(this_frame[line, 5]),
                                                                    int(this_frame[line, 6]), int(this_frame[line, 7]),
                                                                    degree=degree_r)
            angle = degrees(atan(abs(float(Y1) - float(Y2)) / abs(float(X1) - float(X2))))
            one_group_frame=np.vstack((one_group_frame,
                np.array([X1, Y1, X2, Y2, X3, Y3, X4, Y4, angle, this_frame[line, 9],re.findall('(.+?)[0-9]*?.jpg',create_sample[sn])[0]+str(int(max_frame)+sn+1)+'.jpg' ,
                 this_frame[line, 11], this_frame[line, 12], this_frame[line, 13], this_frame[line, 14]])))
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
        write_path=r'E:\frame_rotation\%s%s.jpg' % (re.findall('(.+?)[0-9]*?.jpg',create_sample[sn])[0],str(int(max_frame)+sn+1))
        cv2.imwrite(write_path,img_rotation)
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
    if np.shape(list(set(one_group_frame[:,10])))[0]==8:
        out_add_group=np.vstack((out_add_group,one_group_frame))
    else:
        print('check error%s'%(one_group_frame[0,9]))

#只是添加的frame
np.save(r'.\data\out_add_group',out_add_group)
#######################

all_use_rotation=np.load(r'.\data\all_use_rotation.npy').tolist() #错误，3个坐标
out_add_group=np.load(r'.\data\out_add_group.npy').tolist()

for i in range(1,np.shape(out_add_group)[0]):
    if out_add_group[i] not in all_use_rotation:
        all_use_rotation.append(out_add_group[i])

all_use_rotation.sort(key=lambda l:(l[9],l[10],[14],l[0]))
all_use_rotation=np.array(all_use_rotation)

#全部可抽样数据集，包括了水平框中手动旋转的部分和不够8张帧的
np.save(r'.\data\all_use_add_rotation',all_use_rotation)
