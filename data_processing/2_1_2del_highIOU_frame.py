import numpy as np
import pandas as pd


all_iou=np.load(r'.\data\all_frame_mark_IOU_min.npy')

#计算每个组内有多少帧，帧数小于4 的删除
all_ioupd=pd.DataFrame(all_iou[:,8:])
all_ioupd.columns = ['angle', 'id', 'pic','mp','frame','mark_num','iou','min_frame']

all_ioupd_group_count=all_ioupd.groupby([all_ioupd['id'],all_ioupd['mp'],all_ioupd['min_frame']]).size()


del_pic=[]
for i in range(np.shape(all_iou)[0]):
    if (float(all_iou[i,14])>=0.9 or all_ioupd_group_count.loc[all_iou[i,9]][all_iou[i,11]][all_iou[i,15]]<4) and all_iou[i,10] not in del_pic:
        del_pic.append(all_iou[i,10])

#所有可以用的帧集合
all_use=[]
for i in range(np.shape(all_iou)[0]):
    list=all_iou[i,:13].tolist()
    list.append(all_iou[i,15])
    if all_iou[i,10] not in del_pic and list not in all_use:
        all_use.append(list)

np.save(r'.\data\frame_use_all',all_use)


