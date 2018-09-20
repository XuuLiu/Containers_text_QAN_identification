import numpy as np
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


all_use=np.load(r'.\data\frame_use_all.npy')

#全部的图
all_pic=[]
for i in range(np.shape(all_use)[0]):
    if all_use[i,10] not in all_pic:
        all_pic.append(all_use[i,10])

# 计算帧内每一个标定框与全部标定围成面积的iou，求和，小于阈值的为有错误标定的帧
all_iou=[]

for i in range(np.shape(all_pic)[0]):
    frame=all_use[np.where(all_use[:,10]==all_pic[i])]
    x_list=[]
    y_list=[]
    for j in range(np.shape(frame)[0]):
        x_list.append(int(frame[j, 0]))
        x_list.append(int(frame[j, 2]))
        x_list.append(int(frame[j, 4]))
        x_list.append(int(frame[j, 6]))
        y_list.append(int(frame[j, 1]))
        y_list.append(int(frame[j, 3]))
        y_list.append(int(frame[j, 5]))
        y_list.append(int(frame[j, 7]))

    x_min=min(x_list)
    x_max=max(x_list)
    y_min=min(y_list)
    y_max=max(y_list)

    for j in range(np.shape(frame)[0]):
        xx_min=min([int(frame[j, 0]),int(frame[j, 2]),int(frame[j, 4]),int(frame[j, 6])])
        xx_max=max([int(frame[j, 0]),int(frame[j, 2]),int(frame[j, 4]),int(frame[j, 6])])
        yy_min=min([int(frame[j, 1]),int(frame[j, 3]),int(frame[j, 5]),int(frame[j, 7])])
        yy_max=max([int(frame[j, 1]), int(frame[j, 3]), int(frame[j, 5]), int(frame[j, 7])])
        iou=bb_intersection_over_union([xx_min,yy_min,xx_max,yy_max],[x_min,y_min,x_max,y_max])
        a=frame[j].tolist()
        a.append(iou)
        all_iou.append(a)

all_ioupd=pd.DataFrame(all_iou)
all_ioupd.columns = ['x1','y1','x2','y2','x3','y3','x4','y4','angle', 'id', 'pic','mp', 'frame','min_frame','iou']
all_iou_sum=all_ioupd['iou'].astype(float).groupby(all_ioupd['pic']).sum()
# all_iou_sum.loc['5_01_20180525_183035000.mp4_139.jpg']

#找到所有标记错了的帧
wrong_pic=[]
for i in range(np.shape(all_pic)[0]):
    if all_iou_sum.loc[all_pic[i]]<=0.21:
        wrong_pic.append([all_pic[i],all_iou_sum.loc[all_pic[i]]])
wrong_pic=np.array(wrong_pic)

# 标错的帧中，排列组合计算每两个框的iou
wrong_pic_detail=np.zeros([1,14])
iou_list = []
for i in range(np.shape(wrong_pic)[0]):
    this_pic=all_use[np.where(all_use[:,10]==wrong_pic[i,0])]
    wrong_pic_detail = np.vstack((wrong_pic_detail, this_pic))
    for a in range(np.shape(this_pic)[0]-1):
        for b in range(a+1,np.shape(this_pic)[0]):
            x_list=[int(this_pic[a,0]),int(this_pic[a,2]),int(this_pic[a,4]),int(this_pic[a,6]),int(this_pic[b,0]),int(this_pic[b,2]),int(this_pic[b,4]),int(this_pic[b,6])]
            y_list=[int(this_pic[a,1]),int(this_pic[a,3]),int(this_pic[a,5]),int(this_pic[a,7]),int(this_pic[b,1]),int(this_pic[b,3]),int(this_pic[b,5]),int(this_pic[b,7])]
            x_max=max(x_list)
            x_min=min(x_list)
            y_max=max(y_list)
            y_min=min(y_list)
            iou_a=bb_intersection_over_union([x_min,y_min,x_max,y_max],\
                                           (int(this_pic[a,0]),int(this_pic[a,1]),int(this_pic[a,4]),int(this_pic[a,5])))
            iou_b=bb_intersection_over_union([x_min,y_min,x_max,y_max],\
                                           (int(this_pic[b,0]),int(this_pic[b,1]),int(this_pic[b,4]),int(this_pic[b,5])))

            iou_list.append([this_pic[0,10],this_pic[a,0],this_pic[b,0],this_pic[a,0],iou_a]) #a，b围成的面积中，a框iou
            iou_list.append([this_pic[0, 10], this_pic[a,0], this_pic[b,0], this_pic[b,0], iou_b]) #a，b围成的面积中，b 框iou


#删除每一帧中iou求和最小的框
iou_pd=pd.DataFrame(iou_list)
iou_pd.columns=['pic','a','b','aorb','iou']
iou_sum_pd=iou_pd['iou'].groupby([iou_pd['pic'],iou_pd['aorb']]).sum()

#错误的帧删掉错误的框之后
right_pic_detail=[]
for i in range(1,np.shape(wrong_pic_detail)[0]):
    if not min(iou_sum_pd[wrong_pic_detail[i,10]])==iou_sum_pd[wrong_pic_detail[i,10]][wrong_pic_detail[i,0]]:
        right_pic_detail.append(wrong_pic_detail[i])
right_pic_detail=np.array(right_pic_detail)

all_use=np.load(r'.\data\frame_use_all.npy')
correct_all_use=[]
for i in range(np.shape(all_use)[0]):
    if all_use[i,10] not in right_pic_detail[:,10]:
        correct_all_use.append(all_use[i].tolist())
correct_all_use=np.array(correct_all_use)
correct_all_use=np.vstack((correct_all_use,right_pic_detail))

np.save(r'.\data\correct_all_use',correct_all_use)