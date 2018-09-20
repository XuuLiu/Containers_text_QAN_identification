# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

def condition(condition,iftrue,iffalse):
    if condition:
        return iftrue
    else:
        return iffalse

def get_structure(all_use):
# 为all_use添加水平和竖直的标识，水平1，竖直2
    for i in range(np.shape(all_use)[0]):
        x_len = np.mean([abs(float(all_use[i][2]) - float(all_use[i][0])), abs(float(all_use[i][4]) - float(all_use[i][6]))])
        y_len = np.mean([abs(float(all_use[i][5]) - float(all_use[i][3])), abs(float(all_use[i][7]) - float(all_use[i][1]))])
        all_use[i].append(condition(x_len>y_len,1,2))

    all_use=np.array(all_use)

    #每一个序列的箱号结构，水平or竖直，标定框数

    all_use_pd=pd.DataFrame(all_use)
    all_use_pd.columns = ['x1','y1','x2','y2','x3','y3','x4','y4','angle', 'id', 'pic','mp', 'frame','min_frame','hori1vert2']
    coor_count=all_use_pd.groupby(all_use_pd['pic']).size()

    #pic,水平竖直，标定框个数
    frame_structure=[]
    for i in range(np.shape(all_use)[0]):
        one_frame=[all_use[i,10],all_use[i,14]]
        if one_frame not in frame_structure:
            one_frame.append(coor_count.loc[one_frame[0]])
            frame_structure.append(one_frame)
    frame_structure=np.array(frame_structure)

    #纯序列
    group=[]
    for i in range(np.shape(all_use)[0]):
        l=[all_use[i][9],all_use[i][11],all_use[i][13],all_use[i][14]]
        if l not in group:
            group.append(l)

    # 序列对应第一个帧的结构视为该序列的结构
    for i in range(np.shape(group)[0]):
        try:
            group[i].append(frame_structure[np.where(frame_structure[:,0]==group[i][1]+'.mp4_'+group[i][2]+'.jpg')][0][2])
        except:
            group[i].append(frame_structure[np.where(frame_structure[:,0]==group[i][1]+'.m4_'+group[i][2]+'.jpg')][0][2])


    group_pd=pd.DataFrame(group)
    group_pd.columns=['id','mp4','minframe','hori1vert2','line_num']
    structure_count=group_pd.groupby([group_pd['hori1vert2'],group_pd['line_num']]).size()

    #2/4的实际上就是2/3,特殊箱号，1/4的错了，删除

    structure=[]
    structure.append([1,1,structure_count['1']['1']])
    structure.append([1,2,structure_count['1']['2']])
    structure.append([1,3,structure_count['1']['3']+structure_count['1']['4']])
    structure.append([2,1,structure_count['2']['1']])
    structure.append([2,2,structure_count['2']['2']])

    structure=np.array(structure)
    tot_group=np.shape(group)[0]
    return structure,tot_group

all_use=np.load(r'.\data\correct_all_use.npy').tolist()

structure,tot_group=get_structure(all_use)


#绘图
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
label_list = [u'水平1行', u'水平2行', u'水平3行', u'竖直1列',u'竖直2列']    # 横坐标刻度显示值
rects1 = plt.bar(left=[i for i in range(len(label_list))], height=[round(float(n)/float(tot_group)*100,2) for n in structure[:,2]], width=0.4, alpha=0.8, color=sns.xkcd_rgb['nice blue'])
plt.ylim(0, 60)     # y轴取值范围
plt.ylabel(u"序列占比%",fontproperties=font)
plt.xticks([i for i in range(len(label_list))], label_list,fontproperties=font)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

# 序列内的是否水平、标定框个数
np.save(r'.\data\group_structure.npy',group)