# -*- coding: utf-8 -*-
import numpy as np
import random as rdm
'''
不同箱号结构的调整，让不同格式的箱号的分布比较均匀些
水平1行来自于水平2行
竖直1列来自于竖直2列+自身随机抽样补充（自身抽样补充在2_4_4步中）
竖直2列来自于自身随机抽样补充（自身抽样补充在2_4_4步中）
'''

#增加旋转后的数据
all_use=np.load(r'.\data\all_use_add_rotation.npy').tolist()
#序列的箱号结构
group_structure=np.load(r'.\data\group_structure.npy')

#总数据加上架构信息：14列为1水平2竖直，15列为几行
for i in range(np.shape(all_use)[0]):
    try:
        all_use[i].append(group_structure[np.where(all([all_use[i][9]==group_structure[:,0],all_use[i][11]==group_structure[:,1],all_use[i][13]==group_structure[:,2]])),4][0][0])
    except:
        print(i)


all_use=np.array(all_use)

#将不同格式的箱号分开
base11=all_use[np.where(np.logical_and(all_use[:,14]=='1',all_use[:,15]=='1'))] #水平1行
base12=all_use[np.where(np.logical_and(all_use[:,14]=='1',all_use[:,15]=='2'))] #水平2行
base13=all_use[np.where(np.logical_and(all_use[:,14]=='1',all_use[:,15]=='3'))] #水平3行
base21=all_use[np.where(np.logical_and(all_use[:,14]=='2',all_use[:,15]=='1'))] #竖直1列
base22=all_use[np.where(np.logical_and(all_use[:,14]=='2',all_use[:,15]=='2'))] #竖直2列

group11=group_structure[np.where(np.logical_and(group_structure[:,3]=='1',group_structure[:,4]=='1'))].tolist() #水平1行
group12=group_structure[np.where(np.logical_and(group_structure[:,3]=='1',group_structure[:,4]=='2'))].tolist() #水平2行
group13=group_structure[np.where(np.logical_and(group_structure[:,3]=='1',group_structure[:,4]=='3'))].tolist() #水平3行
group21=group_structure[np.where(np.logical_and(group_structure[:,3]=='2',group_structure[:,4]=='1'))].tolist() #竖直1列
group22=group_structure[np.where(np.logical_and(group_structure[:,3]=='2',group_structure[:,4]=='2'))].tolist() #竖直2列

#计数每种结构有多少个序列
count11=0
count12=0
count13=0
count21=0
count22=0

for i in range(np.shape(group_structure)[0]):
    if group_structure[i,3]=='1'and group_structure[i,4]=='1':
        count11+=1
    elif group_structure[i,3]=='1'and group_structure[i,4]=='2':
        count12+=1
    elif group_structure[i,3]=='1'and group_structure[i,4]=='3':
        count13+=1
    elif group_structure[i,3]=='2'and group_structure[i,4]=='1':
        count21+=1
    elif group_structure[i,3]=='2'and group_structure[i,4]=='2':
        count22+=1
    else:
        pass


#对于水平1行的，从水平2行的抽500-count11个序列
cut_for_11=rdm.sample(group12,500-count11)
cut_for_11_frame=[]
for i in range(np.shape(cut_for_11)[0]):
    this_group=cut_for_11[i]
    this_group_pic=all_use[np.where(logic_and([all_use[:,9]==this_group[0],all_use[:,11]==this_group[1],all_use[:,13]==this_group[2]]))]

    sum_1st_coor=[] #将序列内图片中标记框，x1y1求和，最小的看作为第一行（列）
    for j in range(np.shape(this_group_pic)[0]):
        sum_1st_coor.append([int(this_group_pic[j,0])+int(this_group_pic[j,1]),this_group_pic[j,10]])
    sum_1st_coor=np.array(sum_1st_coor)

    cut_1st_pic=[] #每张图中x1y1加和最小的值
    for k in range(np.shape(sum_1st_coor)[0]):
        one_pic=sum_1st_coor[np.where(sum_1st_coor[:,1]==sum_1st_coor[k,1])]
        pic_1st_line_sum=[sum_1st_coor[k,1],min([int(kk) for kk in one_pic[:,0]])]
        if pic_1st_line_sum not in cut_1st_pic:
            cut_1st_pic.append(pic_1st_line_sum)

    #this_group_cut=[] #该序列内第一列的坐标，注意此处的箱号也对应减少到11位
    for m in range(np.shape(this_group_pic)[0]):
        now_coor=[]
        for n in range(np.shape(cut_1st_pic)[0]):
            if this_group_pic[m,10]==cut_1st_pic[n][0] and int(this_group_pic[m,0])+int(this_group_pic[m,1])==int(cut_1st_pic[n][1]):
                now_coor=this_group_pic[m].copy()
                now_coor[9]=now_coor[9][:11]
                now_coor[15] = '1'
                cut_for_11_frame.append(now_coor.tolist())
    #this_group_cut=np.array(this_group_cut)
cut_for_11_frame=np.array(cut_for_11_frame)
all_11=np.vstack((base11,cut_for_11_frame))


#对于竖直1列的，添加count22个
cut_for_21_frame=[]
for i in range(np.shape(group22)[0]):
    this_group=group22[i]
    this_group_pic=all_use[np.where(logic_and([all_use[:,9]==this_group[0],all_use[:,11]==this_group[1],all_use[:,13]==this_group[2]]))]

    sum_1st_coor=[] #将序列内图片中标记框，x1y1求和，最小的看作为第一行（列）
    for j in range(np.shape(this_group_pic)[0]):
        sum_1st_coor.append([int(this_group_pic[j,0])+int(this_group_pic[j,1]),this_group_pic[j,10]])
    sum_1st_coor=np.array(sum_1st_coor)

    cut_1st_pic=[] #每张图中x1y1加和最小的值
    for k in range(np.shape(sum_1st_coor)[0]):
        one_pic=sum_1st_coor[np.where(sum_1st_coor[:,1]==sum_1st_coor[k,1])]
        pic_1st_line_sum=[sum_1st_coor[k,1],min([int(kk) for kk in one_pic[:,0]])]
        if pic_1st_line_sum not in cut_1st_pic:
            cut_1st_pic.append(pic_1st_line_sum)

    #this_group_cut=[] #该序列内第一列的坐标，注意此处的箱号也对应减少到11位
    for m in range(np.shape(this_group_pic)[0]):
        now_coor=[]
        for n in range(np.shape(cut_1st_pic)[0]):
            if this_group_pic[m,10]==cut_1st_pic[n][0] and int(this_group_pic[m,0])+int(this_group_pic[m,1])==int(cut_1st_pic[n][1]):
                now_coor=this_group_pic[m].copy()
                now_coor[9]=now_coor[9][:11]
                now_coor[15] = '1'
                cut_for_21_frame.append(now_coor.tolist())
    #this_group_cut=np.array(this_group_cut)
cut_for_21_frame=np.array(cut_for_21_frame)
all_21=np.vstack((base21,cut_for_21_frame))

#将不同结构分开保存
np.save(r'.\data\all_11',all_11)
np.save(r'.\data\all_12',base12)
np.save(r'.\data\all_13',base13)
np.save(r'.\data\all_21',all_21)
np.save(r'.\data\all_22',base22)