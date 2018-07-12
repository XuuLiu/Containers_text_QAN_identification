import numpy as np
import pandas as pd
import random as rdm

def condition(condition,iftrue,iffalse):
    if condition:
        return iftrue
    else:
        return iffalse


#采样规则，正序列内，分大角度小角度分层采样
all_use=np.load(r'.\data\frame_use_all.npy').tolist()

# 为all_use添加水平和竖直的标识，水平1，竖直2
for i in range(np.shape(all_use)[0]):
    x_len = np.mean([abs(int(all_use[i][2]) - int(all_use[i][0])), abs(int(all_use[i][4]) - int(all_use[i][6]))])
    y_len = np.mean([abs(int(all_use[i][5]) - int(all_use[i][3])), abs(int(all_use[i][7]) - int(all_use[i][1]))])
    all_use[i].append(condition(x_len>y_len,1,2))

all_use=np.array(all_use)

#单纯的分组key键组合
raw_group=[]
for i in range(np.shape(all_use)[0]):
    x_len=np.mean([abs(int(all_use[i,2])-int(all_use[i,0])),abs(int(all_use[i,4])-int(all_use[i,6]))])
    y_len=np.mean([abs(int(all_use[i,5])-int(all_use[i,3])),abs(int(all_use[i,7])-int(all_use[i,1]))])
    l=[all_use[i,9],all_use[i,11],all_use[i,13],all_use[i,14]]
    if l not in raw_group:
        raw_group.append(l)

#图片整个的倾斜角度看为是其中所有文本框对应角度的平均
all_usepd=pd.DataFrame(all_use)
all_usepd.columns = ['x1','y1','x2','y2','x3','y3','x4','y4','angle', 'id', 'pic','mp', 'frame','min_frame','vertical2']
pic_angel=all_usepd['angle'].astype(int).groupby(all_usepd['pic']).mean()

#分层采样，最大采样4组



choose_file_all=[]
for i in range(np.shape(raw_group)[0]):
    file_group_list=all_use[np.where(np.logical_and(np.logical_and(all_use[:,9]==raw_group[i][0],all_use[:,11]==raw_group[i][1]),\
                                                    all_use[:,14]==raw_group[i][3],all_use[:,13]==raw_group[i][2])),9:][0]
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

    if good_num+bad_num<=8 and good_num+bad_num>=4: #总数小于8则全部采样
        choose_list_file = good_angle_file+bad_angle_file
        rdm.shuffle(choose_list_file)
    elif good_num<=4 and bad_num>=4 and good_num+bad_num>=4: #好样本数小于4，全部采样好样本
        choose_list_file=good_angle_file+rdm.sample(bad_angle_file,8-good_num)
        rdm.shuffle(choose_list_file)
    elif good_num >= 4 and bad_num <= 4 and good_num+bad_num>=4:#坏样本数小于4，全部采样坏样本
        choose_list_file = bad_angle_file + rdm.sample(good_angle_file, 8 - bad_num)
        rdm.shuffle(choose_list_file)
    elif good_num>=16 and bad_num>=16: # 4次
        choose_list_file_all = rdm.sample(good_angle_file, 16) + rdm.sample(bad_angle_file, 16)
        choose_list_file=[]
        l=choose_list_file_all[:4]+choose_list_file_all[-16:-12]
        rdm.shuffle(l)
        choose_list_file.append(l)
        l=choose_list_file_all[4:8]+choose_list_file_all[-12:-8]
        rdm.shuffle(l)
        choose_list_file.append(l)
        l=choose_list_file_all[8:12] + choose_list_file_all[-8:-4]
        rdm.shuffle(l)
        choose_list_file.append(l)
        l=choose_list_file_all[12:16] + choose_list_file_all[-4:]
        rdm.shuffle(l)
        choose_list_file.append(l)
    elif good_num>=12 and bad_num>=12: #3次
        choose_list_file_all = rdm.sample(good_angle_file, 12) + rdm.sample(bad_angle_file, 12)
        choose_list_file=[]
        l=choose_list_file_all[:4]+choose_list_file_all[-12:-8]
        rdm.shuffle(l)
        choose_list_file.append(l)
        l=choose_list_file_all[4:8]+choose_list_file_all[-8:-4]
        rdm.shuffle(l)
        choose_list_file.append(l)
        l=choose_list_file_all[8:12] + choose_list_file_all[-4:]
        rdm.shuffle(l)
        choose_list_file.append(l)
    elif good_num>=8 and bad_num>=8: #2次
        choose_list_file_all = rdm.sample(good_angle_file, 8) + rdm.sample(bad_angle_file, 8)
        choose_list_file=[]
        l=choose_list_file_all[:4]+choose_list_file_all[-8:-4]
        rdm.shuffle(l)
        choose_list_file.append(l)
        l=choose_list_file_all[4:8]+choose_list_file_all[-4:]
        rdm.shuffle(l)
        choose_list_file.append(l)
    elif good_num >= 4 and bad_num >= 4: #1次
        choose_list_file = rdm.sample(good_angle_file, 4) + rdm.sample(bad_angle_file, 4)
        rdm.shuffle(choose_list_file)
    else:
        pass

    if len(choose_list_file)>0:
        choose_file_all.append(choose_list_file)
    else:
        pass

#save
np.save(r'.\data\file_choose2',choose_file_all)

f = open(r'.\data\file_choose2.txt','w')
for one in choose_file_all:
    f.write("%s\n"%one)
f.close()


############################ 补充详细信息
choose_file_all=np.load(r'.\data\file_choose2.npy').tolist() #[group_index][sample_index_in_group][one_frame(8)]
all_use=np.load(r'.\data\frame_use_all.npy')

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

# all_frame_pic[组index][组内采样index][0:8样本小组内的帧index][帧的标记框index][0:13属性] #组内多次采样
# all_frame_pic[组index]               [0:8样本小组内的帧index][帧的标记框index][0:13属性] #组内单次采样

#save
np.save(r'.\data\all_choose2',all_frame_pic)


