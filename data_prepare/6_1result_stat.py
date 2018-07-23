import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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


test_result_all=np.load(r'.\data\test_result_all.npy')


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
sns.distplot(best_scores, kde=False, bins=20, rug=False)
plot_percentile(best_scores,20,50)
plot_percentile(best_scores,20,5)
plot_percentile(best_scores,20,95)


#低质量帧统计

low_frame_score=[]
for i in range(np.shape(test_result_all)[0]):
    if float(test_result_all[i,5])*float(test_result_all[i,6])==0:
        low_frame_score.append(float(test_result_all[i,7]))

np.percentile(low_frame_score,50)
sns.distplot(low_frame_score, kde=False, bins=20, rug=False)
plot_percentile(low_frame_score,20,50)
plot_percentile(low_frame_score,20,5)
plot_percentile(low_frame_score,20,95)

# 每一序列中前两个帧的评分
group=list(set(test_result_all[:,0]))

top_2=[]
for one_group in group:
    this_group_score=test_result_all[np.where(test_result_all[:,0]==one_group),6:9][0]
    score=[float(s) for s in this_group_score[:,2]]
    score.sort()
    top1=score[-1]
    top2=score[-2]
    top_2.append(this_group_score[np.where(this_group_score[:,2]==str(top1))][0].tolist())
    top_2.append(this_group_score[np.where(this_group_score[:, 2] == str(top2))][0].tolist())
top_2=np.array(top_2)

# 绘图
top_2_score=[float(score) for score in top_2[:,2]]
sns.distplot(top_2_score, kde=False, bins=20, rug=False)
plot_percentile(top_2_score,20,50)
plot_percentile(top_2_score,20,5)
plot_percentile(top_2_score,20,95)

# 看01比例
same=0
different=0
ambiguous=0
for i in range(np.shape(top_2)[0]):
    if float(top_2[i,0])*float(top_2[i,1])==1:
        same+=1
    elif float(top_2[i,0])*float(top_2[i,1])==0:
        different+=1
    else:
        ambiguous+=1
print('same_count:%d   different_count:%d    ambiguous_count:%d'%(same,different,ambiguous))

# 阈值选择

threshold=0.45
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
