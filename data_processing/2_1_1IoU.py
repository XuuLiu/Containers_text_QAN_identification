import numpy as np
import math
import re
import pandas as pd

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

# 读取每个框的坐标
stat=np.load(r'.\data\16_20180530_1_angle.npy')
for i in range(2,9):
    statone=np.load(r'.\data\16_20180530_%s_angle.npy'%i)
    stat=np.vstack((stat,statone))

for i in range(1,11):
    statone=np.load(r'.\data\17_20180606_%s_angle.npy'%i)
    stat=np.vstack((stat,statone))
for i in range(1,20):
    statone=np.load(r'.\data\18_20180608_%s_angle.npy'%i)
    stat=np.vstack((stat,statone))
basic_info=stat.tolist()

# 标记框的相对位置，用x1来sort
basic_info.sort(key=lambda l:(l[9],l[10],l[0],l[1],l[4],l[5]))
for i in range(np.shape(basic_info)[0]):
    basic_info[i].append(re.findall(r'(.+?).mp?4_', basic_info[i][10])[0]) #index11 mp4
    basic_info[i].append(int(re.findall(r'mp?4_(.+?).jpg',basic_info[i][10])[0])) #index 12 frame

basic_infopd=pd.DataFrame(basic_info)
basic_infopd.columns = ['x1', 'y1', 'x2', 'y2', 'x3','y3','x4', 'y4', 'angle', 'id', 'pic','mp','frame']
basic_infopd['group_sort']=basic_infopd[['x1']].astype(int).groupby([basic_infopd['id'],basic_infopd['mp'],basic_infopd['frame']]).rank(ascending=0,method='dense')


#计算框对应与上一帧的IoU
iou_list=[]
for i in range(np.shape(basic_infopd)[0]):
    iou=0
    if i<=3:
        for lag in range(i):
            if basic_infopd.iloc[i]['id']==basic_infopd.iloc[i-lag]['id'] and basic_infopd.iloc[i]['mp']==basic_infopd.iloc[i-lag]['mp'] \
                    and basic_infopd.iloc[i]['group_sort'] == basic_infopd.iloc[i - lag]['group_sort'] and basic_infopd.iloc[i]['frame']-1==basic_infopd.iloc[i-lag]['frame']:
                iou=bb_intersection_over_union([int(basic_infopd.iloc[i-lag]['x1']), int(basic_infopd.iloc[i-lag]['y1']), int(basic_infopd.iloc[i-lag]['x3']), int(basic_infopd.iloc[i-lag]['y3'])],
                           [int(basic_infopd.iloc[i]['x1']), int(basic_infopd.iloc[i]['y1']), int(basic_infopd.iloc[i]['x3']), int(basic_infopd.iloc[i]['y3'])])
                break
    else:
        for lag in range(3):
            if basic_infopd.iloc[i]['id']==basic_infopd.iloc[i-lag]['id'] and basic_infopd.iloc[i]['mp']==basic_infopd.iloc[i-lag]['mp'] \
                    and basic_infopd.iloc[i]['group_sort'] == basic_infopd.iloc[i - lag]['group_sort'] and basic_infopd.iloc[i]['frame']-1==basic_infopd.iloc[i-lag]['frame']:
                iou=bb_intersection_over_union([int(basic_infopd.iloc[i-lag]['x1']), int(basic_infopd.iloc[i-lag]['y1']), int(basic_infopd.iloc[i-lag]['x3']), int(basic_infopd.iloc[i-lag]['y3'])],
                           [int(basic_infopd.iloc[i]['x1']), int(basic_infopd.iloc[i]['y1']), int(basic_infopd.iloc[i]['x3']), int(basic_infopd.iloc[i]['y3'])])
                break
    iou_list.append(iou)

basic_infopd.insert(14,'iou', iou_list)
np.save(r'.\data\all_frame_mark_IOU',basic_infopd)


# 分组的minframe
basic_infopd=pd.DataFrame(np.load(r'.\data\all_frame_mark_IOU.npy'))
basic_infopd.columns = ['x1', 'y1', 'x2', 'y2', 'x3','y3','x4', 'y4', 'angle', 'id', 'pic','mp','frame','rank_in_group','iou']
group=np.load(r'.\data\angle_minframe_group.npy')
min_frame=[]

for i in range(np.shape(basic_infopd)[0]):
    min_frame.append(group[np.where(group[:, 1] == basic_infopd.iloc[i]['pic'])][0][5])

basic_infopd.insert(15,'minframe', min_frame)

#所有的帧和其上一帧标记框的IOU
basic_infopd.to_csv(r'./data/angle_minframe_group.csv')
np.save(r'.\data\all_frame_mark_IOU_min',basic_infopd)


'''
i=21280
bb_intersection_over_union([int(basic_info[i-2][0]), int(basic_info[i-2][1]), int(basic_info[i-2][4]), int(basic_info[i-2][5])],
                           [int(basic_info[i][0]), int(basic_info[i][1]), int(basic_info[i][4]), int(basic_info[i][5])])




for i in range(np.shape(basic_info)[0]):
    if basic_info[i][-1]=='2_01_20180525_163802000.mp4_458.jpg':
        print(i)
'''