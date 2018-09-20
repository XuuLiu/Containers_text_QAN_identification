import numpy as np
import pandas as pd

'''
计算两行的箱号，第一行切割的范围和第二行的重合IoU，不希望补充1行的情况下，第二行产生干扰
'''

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




all_use = np.load(r'.\data\all_use_add_rotation.npy')
all_use_pd=pd.DataFrame(all_use)
all_use_pd.columns = ['x1','y1','x2','y2','x3','y3','x4','y4','angle', 'id', 'pic','mp', 'frame','min_frame','vertical2']
frame_count=all_use_pd.groupby(all_use_pd['pic']).size()

#两行的
pic_need_cal=[]
for i in range(np.shape(all_use)[0]):
    if frame_count[all_use[i,10]]==2:
        pic_need_cal.append(all_use[i].tolist())

pic_need_cal=np.array(pic_need_cal)

pic=list(set(pic_need_cal[:,10]))

pic_iou=[]
for one in pic:
    pic_loc=pic_need_cal[np.where(pic_need_cal[:,10]==one)]
    line1_sum=min([int(pic_loc[0,0])+int(pic_loc[0,1]),int(pic_loc[1,0])+int(pic_loc[1,1])])
    if int(pic_loc[0,0])+int(pic_loc[0,1])==line1_sum:
        line1=pic_loc[0]
        line2=pic_loc[1]
    else:
        line1=pic_loc[1]
        line2=pic_loc[0]
    cut_loc=[
        min([int(line1[0]),int(line1[2]),int(line1[4]),int(line1[6])]),
        min([int(line1[1]),int(line1[3]),int(line1[5]),int(line1[7])]),
        max([int(line1[0]), int(line1[2]), int(line1[4]), int(line1[6])]),
        max([int(line1[1]), int(line1[3]), int(line1[5]), int(line1[7])])
    ]
    line2_loc=[
        min([int(line2[0]), int(line2[2]), int(line2[4]), int(line2[6])]),
        min([int(line2[1]), int(line2[3]), int(line2[5]), int(line2[7])]),
        max([int(line2[0]), int(line2[2]), int(line2[4]), int(line2[6])]),
        max([int(line2[1]), int(line2[3]), int(line2[5]), int(line2[7])])

    ]
    pic_iou.append([one,bb_intersection_over_union(cut_loc,line2_loc)])


pic_iou=np.array(pic_iou)
# 选择阈值为<0.07
np.save(r'.\data\two_line_iou',pic_iou)