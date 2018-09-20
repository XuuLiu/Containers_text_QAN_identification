# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import math
import re


def condition(condition,iftrue,iffalse):
    if condition:
        return iftrue
    else:
        return iffalse

def cal_angle(coor,way):
    # coor is x1,y1,x2,y2,x3,y3,x4,y4
    # way is '水平' or '垂直'
    #coor=all_recog[1][4:12]
    if way=='水平':
        angle=math.degrees(math.atan(abs(float(coor[1])-float(coor[3]))/abs(float(coor[0])-float(coor[2]))))
        return str(int(angle))
    #coor=all_recog[7][4:12]
    else:
        angle=math.degrees(math.atan(abs(float(coor[0])-float(coor[6]))/abs(float(coor[1])-float(coor[7]))))
        if angle>45:
            return 90-int(angle)
        else:
            return int(angle)


def get_angle(filename,filenumber):
    for n in range(1, filenumber + 1):
        #n=1
        f = open(r".\data\%s_%s.txt" % (filename, n))
        # f = open("16_20180530_%s.txt"%n)
        all_raw = []
        line = f.readline().replace('\xef', '').replace('\xbb', '').replace('\xbf', '').replace('\n', '').replace('  ', ' ')
        while line:
            a = line.split(' ')
            all_raw.append(a)
            line = f.readline().replace('\xef', '').replace('\xbb', '').replace('\xbf', '').replace('\n', '').replace('  ',
                                                                                                                      ' ')
        f.close()

        # pic have tag
        all_recog = []
        for i in range(len(all_raw)):
            if all_raw[i][1] != '0':
                all_recog.append(all_raw[i])

        # get coordinates
        all_angle=[]
        for i in range(np.shape(all_recog)[0]):
            if len(all_recog[i])>2:
                if all_recog[i-1][1]=='1':
                    coor_num=(len(all_recog[i])-5)/8
                    coor=[]
                    for j in range(coor_num):
                        coor.append(all_recog[i][4+8*j:4+8*(j+1)])
                    for j in range(coor_num):
                        angle=cal_angle(coor[j],all_recog[i][1])
                        coor[j].append(angle)
                        coor[j].append(all_recog[i][2])
                        coor[j].append(all_recog[i-1][0])
                        all_angle.append(coor[j])

        all_angle=np.array(all_angle)

        np.save(r'.\data\%s_%s_angle'%(filename,n),all_angle)

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
    return group,structure,tot_group



#读取测试数据文件
#get_angle('19_20180614_2',14)

#read all data
stat=np.load(r'.\data\19_20180614_2_1_angle.npy')
for i in range(2,14):
    statone=np.load(r'.\data\19_20180614_2_%s_angle.npy'%i)
    stat=np.vstack((stat,statone))

#添加分组key键
stat=stat.tolist()
for i in range(np.shape(stat)[0]):
    stat[i].insert(11,int(re.findall(r'mp?4_(.+?).jpg',stat[i][10])[0])) #本帧
    stat[i].insert(11, re.findall(r'(.+?).mp?4_', stat[i][10])[0]) #本mp4

#计算每个标定框的倾斜角度
stat.sort(key=lambda l:(l[9],l[11],l[12]))
stat[0].append(stat[0][12]) #第一个的最小帧
for i in range(1,np.shape(stat)[0]):
    if stat[i][9]==stat[i-1][9] and stat[i][11]==stat[i-1][11] and stat[i][12]-stat[i-1][12]<100: #暂定100
        stat[i].append(stat[i-1][13])
    else:
        stat[i].append(stat[i][12])

np.save(r'.\data\6_0_1stat',stat)

#################################以下为箱号格式的统计
'''
#序列的箱号格式
group_structure,structure_count,tot_group=get_structure(stat)
np.save(r'.\data\6_0_1test_group_structure',group_structure)

#绘制分布
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
label_list = [u'水平1行', u'水平2行', u'水平3行', u'竖直1列',u'竖直2列']    # 横坐标刻度显示值
rects1 = plt.bar(left=[i for i in range(len(label_list))], height=[round(float(n)/float(tot_group)*100,2) for n in structure_count[:,2]], width=0.4, alpha=0.8, color=sns.xkcd_rgb['nice blue'])
plt.ylim(0, 60)     # y轴取值范围
plt.ylabel(u"序列占比%",fontproperties=font)
plt.xticks([i for i in range(len(label_list))], label_list,fontproperties=font)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")


###############################
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
label_list = [u'水平1行', u'水平2行', u'水平3行', u'竖直1列',u'竖直2列']    # 横坐标刻度显示值
rects1 = plt.bar(left=[i for i in range(len(label_list))], height=[0.54,0.73,0.39,0.30,0.54], width=0.4, alpha=0.8, color=sns.xkcd_rgb['nice blue'])
plt.ylim(0, 0.8)     # y轴取值范围
plt.ylabel(u"不同结构的箱号对应网络评分均值",fontproperties=font)
plt.xticks([i for i in range(len(label_list))], label_list,fontproperties=font)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

'''