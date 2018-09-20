import numpy as np
import pandas as pd
import random as rdm
'''
根据不同结构的箱号进行分层抽样
先选哪些序列
再选序列类哪些图，图要满足第二行不会进入到剪切的范围内
'''
def logic_and(list):
    result=list[0]
    for i in range(1,np.shape(list)[0]):
        result=np.logical_and(list[i],result)
    return result

def get_group_index(data):
    # data=all_11
    group=[]
    for i in range(np.shape(data)[0]):
        l=[data[i,9],data[i,11],data[i,13],data[i,14],data[i,15]]
        if l not in group:
            group.append(l)
    out_group=[]
    for i in range(np.shape(group)[0]):
        this_group=data[np.where(logic_and([group[i][0]==data[:,9],group[i][1]==data[:,11],group[i][2]==data[:,13],group[i][3]==data[:,14],group[i][4]==data[:,15]]))]
        pic_name = list(set([t for t in this_group[:, 10]]))
        if np.shape(pic_name)[0]<8:
            pass
        else:
            group[i].append(np.shape(this_group)[0])
            out_group.append(group[i])
    return out_group

def random_file_choose_method(good_pic,bad_pic,choose_good_num=4,choose_tot_num=8):
    # 分层高低质量帧，随机选择样本。choose_good_num为选择高质量帧的个数，choose_tot_num为采样的总数
    good_num=np.shape(good_pic)[0]
    bad_num=np.shape(bad_pic)[0]

    choose_list_file=[]
    if good_num + bad_num == choose_tot_num:  # 总数为8则全部采样
        choose_list_file = good_pic + bad_pic
        rdm.shuffle(choose_list_file)
    elif good_num <= choose_good_num and bad_num >= choose_tot_num-choose_good_num and good_num + bad_num >= choose_tot_num:  # 好样本数小于4，全部采样好样本
        choose_list_file = good_pic + rdm.sample(bad_pic, choose_tot_num - good_num)
        rdm.shuffle(choose_list_file)
    elif good_num >= choose_good_num and bad_num <= choose_tot_num-choose_good_num and good_num + bad_num >= choose_tot_num:  # 坏样本数小于4，全部采样坏样本
        choose_list_file = bad_pic + rdm.sample(good_pic, choose_tot_num - bad_num)
        rdm.shuffle(choose_list_file)
    elif good_num >= choose_good_num and bad_num >= choose_tot_num-choose_good_num:  # 1次
        choose_list_file = rdm.sample(good_pic, choose_good_num) + rdm.sample(bad_pic, choose_tot_num-choose_good_num)
        rdm.shuffle(choose_list_file)
    else:
        pass
    return choose_list_file

def random_sample_from_group(data,group,is_need_iou):
    # data=all_12;group=group12
    #分层随机抽样，每一个序列内抽3组（44,35，53）
    choose_file_all=[]
    for i in range(np.shape(group)[0]):

        # 该序列内的全部图片
        this_group_pic = data[np.where(logic_and([group[i][0] == data[:, 9], group[i][1] == data[:, 11],group[i][2] == data[:, 13], group[i][3] == data[:, 14], group[i][4] == data[:, 15]]))]

        #图片整个的倾斜角度看为是其中所有文本框对应角度的平均
        this_group_pic_pd=pd.DataFrame(this_group_pic)
        this_group_pic_pd.columns = ['x1','y1','x2','y2','x3','y3','x4','y4','angle', 'id', 'pic','mp', 'frame','min_frame','vertical2','line_num']
        pic_angel=this_group_pic_pd['angle'].astype(float).astype(int).groupby(this_group_pic_pd['pic']).mean()

        pic_name = list(set([t for t in this_group_pic[:,10]]))

        #依据角度分高低质量帧
        good_pic=[]
        bad_pic=[]

        for j in range(np.shape(pic_name)[0]):
            if is_need_iou==1:
                try:
                    if float(pic_iou[np.where(pic_iou[:,0]==pic_name[j]),1])<0.07:
                        if pic_angel['%s'%pic_name[j]]<9:
                            good_pic.append(pic_name[j])
                        elif pic_angel['%s'%pic_name[j]]>=9:
                            bad_pic.append(pic_name[j])
                except:
                    if pic_angel['%s' % pic_name[j]] < 9:
                        good_pic.append(pic_name[j])
                    elif pic_angel['%s' % pic_name[j]] >= 9:
                        bad_pic.append(pic_name[j])
            else:
                if pic_angel['%s' % pic_name[j]] < 9:
                    good_pic.append(pic_name[j])
                elif pic_angel['%s' % pic_name[j]] >= 9:
                    bad_pic.append(pic_name[j])

        # 高低质量帧分层抽样
        choose_list_file1=random_file_choose_method(good_pic, bad_pic, choose_good_num=4, choose_tot_num=8)
        choose_list_file2=random_file_choose_method(good_pic, bad_pic, choose_good_num=6, choose_tot_num=8)
        choose_list_file3=random_file_choose_method(good_pic, bad_pic, choose_good_num=5, choose_tot_num=8)

        if len(choose_list_file1) > 0:
            choose_file_all.append(choose_list_file1)
        else:
            print(i)
        if len(choose_list_file2) > 0:
            choose_file_all.append(choose_list_file2)
        else:
            print(i)
        if len(choose_list_file3) > 0:
            choose_file_all.append(choose_list_file3)
        else:
            print(i)
    return choose_file_all

# 两行两列的两个标定框裁剪的iou
pic_iou=np.load(r'.\data\two_line_iou.npy')

all_11=np.load(r'.\data\all_11.npy')
all_12=np.load(r'.\data\all_12.npy')
all_13=np.load(r'.\data\all_13.npy')
all_21=np.load(r'.\data\all_21.npy')
all_22=np.load(r'.\data\all_22.npy')
all_use=np.vstack((all_11,all_12,all_13,all_21,all_22))
np.save(r'.\data\all_use_v822',all_use)

#水平1行的直接有多少个组抽多少个就可以
#水平2行的直接有多少个组抽多少个就可以
#水平3行的直接有多少个组抽多少个就可以
#竖直1列的抽全部组，并重复抽100个组
#竖直2列的抽全部组*2

#确定每个结构中有哪些组[9,11,13,14,15]
group11=get_group_index(all_11)
group12=get_group_index(all_12)
group13=get_group_index(all_13)
group21=get_group_index(all_21)
group22=get_group_index(all_22)


#####直接有多少个组抽多少个就可以的情况，还需要对组抽样的情况
choose_file_11=random_sample_from_group(data=all_11,group=group11,is_need_iou=1)
choose_file_11=np.array(choose_file_11)
choose_file_12=random_sample_from_group(data=all_12,group=group12,is_need_iou=0)
choose_file_12=np.array(choose_file_12)
choose_file_13=random_sample_from_group(data=all_13,group=group13,is_need_iou=0)
choose_file_13=np.array(choose_file_13)

#竖直1列的抽全部组，并重复抽100个组
choose_group_21=group21+rdm.sample(group21,100)
choose_file_21=random_sample_from_group(data=all_21,group=choose_group_21,is_need_iou=1)
choose_file_21=np.array(choose_file_21)

#竖直2列的抽全部组*2
choose_group_22=group22+group22
choose_file_22=random_sample_from_group(data=all_22,group=choose_group_22,is_need_iou=0)
choose_file_22=np.array(choose_file_22)

#全部的抽样组合

np.save(r'.\data\all_choose_file_11',choose_file_11)
np.save(r'.\data\all_choose_file_12',choose_file_12)
np.save(r'.\data\all_choose_file_13',choose_file_13)
np.save(r'.\data\all_choose_file_21',choose_file_21)
np.save(r'.\data\all_choose_file_22',choose_file_22)

###############################补充抽样图片的标定信息
choose_file_11=np.load(r'.\data\all_choose_file_11.npy')
choose_file_12=np.load(r'.\data\all_choose_file_12.npy')
choose_file_13=np.load(r'.\data\all_choose_file_13.npy')
choose_file_21=np.load(r'.\data\all_choose_file_21.npy')
choose_file_22=np.load(r'.\data\all_choose_file_22.npy')

all_use=np.load(r'.\data\all_use_v822.npy')


all_frame_pic=[]

for one_group in choose_file_11:
    frame_pic=[]
    for one_pic in one_group:
        frame_pic.append(all_use[np.where(logic_and([all_use[:,14]=='1',all_use[:,15]=='1',all_use[:,10]==one_pic]))].tolist())
    all_frame_pic.append(frame_pic)

for one_group in choose_file_12:
    frame_pic=[]
    for one_pic in one_group:
        frame_pic.append(all_use[np.where(logic_and([all_use[:,14]=='1',all_use[:,15]=='2',all_use[:,10]==one_pic]))].tolist())
    all_frame_pic.append(frame_pic)

for one_group in choose_file_13:
    frame_pic=[]
    for one_pic in one_group:
        frame_pic.append(all_use[np.where(logic_and([all_use[:,14]=='1',all_use[:,15]=='3',all_use[:,10]==one_pic]))].tolist())
    all_frame_pic.append(frame_pic)

for one_group in choose_file_21:
    frame_pic = []
    for one_pic in one_group:
        frame_pic.append(all_use[np.where(logic_and([all_use[:, 14] == '2', all_use[:, 15] == '1', all_use[:, 10] == one_pic]))].tolist())
    all_frame_pic.append(frame_pic)

for one_group in choose_file_22:
    frame_pic = []
    for one_pic in one_group:
        frame_pic.append(all_use[np.where(logic_and([all_use[:, 14] == '2', all_use[:, 15] == '2', all_use[:, 10] == one_pic]))].tolist())
    all_frame_pic.append(frame_pic)


# [序列index][:7序列内帧index][帧内标定框index][]

np.save(r'.\data\all_choose_v822',all_frame_pic)


