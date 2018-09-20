import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import cv2
import re
import pandas as pd

def plot_percentile(data,bins_num,per):
    hist, bins = np.histogram(data, bins=bins_num,)
    x=np.percentile(data,per)
    for i in range(np.shape(bins)[0]-1):
        if bins[i]<=x and bins[i+1]>=x:
            index=i
    y=hist[index]
    plt.plot([x,x,],[0,y],'k--',linewidth=1.5,c=sns.xkcd_rgb['nice blue'])
    plt.text(x+0.01,1,str(round(x,2)),color=sns.xkcd_rgb['nice blue'])
    plt.text(x-(bins[1]-bins[0]),y+2,str(per)+'%',color=sns.xkcd_rgb['nice blue'])

def plot_value(data, bins_num):
    # data=count_large[:,0]
    # bins_num=5
    hist, bins = np.histogram(data, bins=bins_num, )
    tot = sum(hist)
    for i in range(bins_num):
        plt.text(bins[i], hist[i] + 2, str(round(float(hist[i]) / float(tot), 4) * 100) + '%',
                 color=sns.xkcd_rgb['nice blue'])

def logic_and(list):
    result=list[0]
    for i in range(1,np.shape(list)[0]):
        result=np.logical_and(list[i],result)
    return result

#统计测试结果

test_result_all=np.load(r'.\data\test_result_all_pool4_v822.npy').tolist()

#########################每种箱号结构的分数统计
#测试数据集的序列箱号结构
group_structure=np.load(r'.\data\6_0_1test_group_structure.npy')

for i in range(np.shape(test_result_all)[0]):
    test_result_all[i].append(group_structure[np.where(logic_and([group_structure[:,0]==test_result_all[i][0].split('_')[0], \
                                                                  group_structure[:, 2] ==test_result_all[i][0].split('_')[-1], \
                                                                 group_structure[:, 1] ==re.findall('^[A-Z0-9a-z]+_(.*?)_[0-9]+$',test_result_all[i][0])])),3][0][0])
    test_result_all[i].append(group_structure[np.where(logic_and([group_structure[:,0]==test_result_all[i][0].split('_')[0], \
                                                                  group_structure[:, 2] ==test_result_all[i][0].split('_')[-1], \
                                                                 group_structure[:, 1] ==re.findall('^[A-Z0-9a-z]+_(.*?)_[0-9]+$',test_result_all[i][0])])),4][0][0])



test_result_all_pd=pd.DataFrame(test_result_all)
test_result_all_pd.columns=['group','id','pic','min_frame','mp4','angle','angle_score','rengong_score','network_score','max_score_ingroup','hori1verti2','textline_num']
structure_avg=test_result_all_pd['network_score'].astype(float).groupby([test_result_all_pd['hori1verti2'],test_result_all_pd['textline_num']]).mean()
structure_max_avg=test_result_all_pd['max_score_ingroup'].astype(float).groupby([test_result_all_pd['hori1verti2'],test_result_all_pd['textline_num']]).mean()


#均值统计
test_structure_mean_stat=[]
test_structure_mean_stat.append([1,1,structure_avg['1']['1']])
test_structure_mean_stat.append([1,2,structure_avg['1']['2']])
test_structure_mean_stat.append([1,3,structure_avg['1']['3']])
test_structure_mean_stat.append([2,1,structure_avg['2']['1']])
test_structure_mean_stat.append([2,2,structure_avg['2']['2']])
test_structure_mean_stat=np.array(test_structure_mean_stat)

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
label_list = [u'水平1行', u'水平2行', u'水平3行', u'竖直1列', u'竖直2列']  # 横坐标刻度显示值
rects1 = plt.bar(left=[i for i in range(len(label_list))],
                 height=[round(float(n), 2) for n in test_structure_mean_stat[:,2]], width=0.4, alpha=0.8,
                 color=sns.xkcd_rgb['nice blue'])
plt.ylim(0, 0.8)  # y轴取值范围
plt.ylabel(u"不同结构的箱号对应网络评分均值", fontproperties=font)
plt.xticks([i for i in range(len(label_list))], label_list, fontproperties=font)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + 0.01, str(height), ha="center", va="bottom")

#########################每种箱号结构的人工分数统计

test_result_all_pd['rengong_score_tot']=test_result_all_pd['rengong_score'].astype(float)*test_result_all_pd['angle_score'].astype(float)
structure_rengong_avg=test_result_all_pd['rengong_score_tot'].astype(float).groupby([test_result_all_pd['hori1verti2'],test_result_all_pd['textline_num']]).mean()

test_structure_mean_stat=[]
test_structure_mean_stat.append([1,1,structure_rengong_avg['1']['1']])
test_structure_mean_stat.append([1,2,structure_rengong_avg['1']['2']])
test_structure_mean_stat.append([1,3,structure_rengong_avg['1']['3']])
test_structure_mean_stat.append([2,1,structure_rengong_avg['2']['1']])
test_structure_mean_stat.append([2,2,structure_rengong_avg['2']['2']])
test_structure_mean_stat=np.array(test_structure_mean_stat)

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
label_list = [u'水平1行', u'水平2行', u'水平3行', u'竖直1列', u'竖直2列']  # 横坐标刻度显示值
rects1 = plt.bar(left=[i for i in range(len(label_list))],
                 height=[round(float(n), 2) for n in test_structure_mean_stat[:,2]], width=0.4, alpha=0.8,
                 color=sns.xkcd_rgb['nice blue'])
plt.ylim(0, 1.1)  # y轴取值范围
plt.ylabel(u"不同结构的箱号对应人工评分均值", fontproperties=font)
plt.xticks([i for i in range(len(label_list))], label_list, fontproperties=font)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + 0.01, str(height), ha="center", va="bottom")

#####################################
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
label_list = [u'水平1行', u'水平2行', u'水平3行', u'竖直1列', u'竖直2列']  # 横坐标刻度显示值
rects1 = plt.bar(left=[i for i in range(len(label_list))],
                 height=[0.59, 0.73, 0.48, 0.42, 0.59], width=0.4, alpha=0.8,
                 color=sns.xkcd_rgb['nice blue'])
plt.ylim(0, 0.8)  # y轴取值范围
plt.ylabel(u"不同结构的箱号对应人工评分均值", fontproperties=font)
plt.xticks([i for i in range(len(label_list))], label_list, fontproperties=font)
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + 0.01, str(height), ha="center", va="bottom")





######################################################################统计图
# 最优帧统计

best_frame_score=[]
for i in range(np.shape(test_result_all)[0]):
    best=[test_result_all[i,0],test_result_all[i,2],test_result_all[i,3],test_result_all[i,9]]
    if best not in best_frame_score:
        best_frame_score.append(best)
    else:
        pass
best_frame_score=np.array(best_frame_score)
best_scores=[float(score) for score in best_frame_score[:,3]]


bin_num=5
sns.distplot(best_scores, kde=False, bins=bin_num, rug=False)
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.xlabel(u'最优帧评分',fontproperties=font)
plt.ylabel(u'落入x区间的序列个数',fontproperties=font)
plot_value(best_scores,bin_num)

#高质量帧统计

high_frame_score=[]
for i in range(np.shape(test_result_all)[0]):
    if float(test_result_all[i][6])*float(test_result_all[i][7])==1:
        high_frame_score.append(float(test_result_all[i][8]))

bin_num=5
sns.distplot(high_frame_score, kde=False, bins=bin_num, rug=False)
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.xlabel(u'人工判定的高质量帧评分',fontproperties=font)
plt.ylabel(u'落入x区间的帧个数',fontproperties=font)
plot_value(high_frame_score,bin_num)


#低质量帧统计

low_frame_score=[]
for i in range(np.shape(test_result_all)[0]):
    if float(test_result_all[i,6])*float(test_result_all[i,7])==0:
        low_frame_score.append(float(test_result_all[i,8]))

bin_num=5
sns.distplot(low_frame_score, kde=False, bins=bin_num, rug=False)
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.xlabel(u'人工判定的低质量帧评分',fontproperties=font)
plt.ylabel(u'落入x区间的帧个数',fontproperties=font)
plot_value(low_frame_score,bin_num)


#中等质量帧统计

medium_frame_score=[]
for i in range(np.shape(test_result_all)[0]):
    if float(test_result_all[i][6])*float(test_result_all[i][7])!=0 and float(test_result_all[i][6])*float(test_result_all[i][7])!=1:
        medium_frame_score.append(float(test_result_all[i][8]))

bin_num=5
sns.distplot(medium_frame_score, kde=False, bins=bin_num, rug=False)
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.xlabel(u'人工判定的中等质量帧评分',fontproperties=font)
plt.ylabel(u'落入x区间的帧个数',fontproperties=font)
plot_value(medium_frame_score,bin_num)

###########################角度与网络评分6，7
test_result_all=np.load(r'.\data\test_result_all_pool4_v822.npy')
angle_0=np.mean([float(n) for n in test_result_all[np.where(test_result_all[:,6]=='0'),8][0]])
angle_1=np.mean([float(n) for n in test_result_all[np.where(test_result_all[:,6]=='1'),8][0]])
angle_o=np.mean([float(n) for n in test_result_all[np.where(np.logical_and(test_result_all[:,6]!='1',test_result_all[:,6]!='0')),8][0]])

other_0=np.mean([float(n) for n in test_result_all[np.where(test_result_all[:,7]=='0'),8][0]])
other_1=np.mean([float(n) for n in test_result_all[np.where(test_result_all[:,7]=='1'),8][0]])
other_o=np.mean([float(n) for n in test_result_all[np.where(np.logical_and(test_result_all[:,7]!='1',test_result_all[:,7]!='0')),8][0]])


#########################统计网络判定和人工判定的符合度

test_result_all=np.load(r'.\data\test_result_all_pool4_v822.npy')
same=0
different=0
ambiguous=0
different_list=[]
ambiguous_list=[]
for i in range(np.shape(test_result_all)[0]):
    if float(test_result_all[i,8])==float(test_result_all[i,9]):
        if float(test_result_all[i,6])*float(test_result_all[i,7])==1:
            same+=1
        elif float(test_result_all[i,6])*float(test_result_all[i,7])==0:
            different+=1
            different_list.append(test_result_all[i].tolist())
        else:
            ambiguous += 1
            ambiguous_list.append(test_result_all[i].tolist())
    else:
        pass
tot=same+different+ambiguous
print('same_count:%4f   different_count:%4f    ambiguous_count:%4f'%(float(same)/float(tot),float(different)/float(tot),float(ambiguous)/float(tot)))
different_list=np.array(different_list)
ambiguous_list=np.array(ambiguous_list)

##########################
for pic in ambiguous_list[:,0]:
    img=cv2.imread(r'..\test_result_pool4_v822\%s.jpg'%(pic))
    cv2.imwrite(r'..\test_wrong\%s.jpg'%(pic),img)


############################################## 阈值选择

test_result_all=np.load(r'.\data\test_result_all_pool4_v822.npy')


threshold=0.55
all_score=test_result_all[:,6:9]
low=0
high=0
medium=0
tot_high=0
tot_low=0
tot_medium=0
for i in range(np.shape(all_score)[0]):
    if float(all_score[i, 0]) * float(all_score[i, 1]) == 0:
        tot_low+=1
        if float(all_score[i,2])>=threshold:
            low+=1

    elif float(all_score[i,0])*float(all_score[i,1])!=0 and  float(all_score[i,0])*float(all_score[i,1])!=1:
        tot_medium += 1
        if float(all_score[i, 2]) >= threshold:
            medium += 1

    else:
        tot_high+=1
        if float(all_score[i, 2]) < threshold:
            high+=1


print('low:%4d%%   medium:%4d%%    high:%4d%%'%(float(low)/float(tot_low)*100,float(medium)/float(tot_medium)*100,float(high)/float(tot_high)*100))
