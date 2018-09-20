import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib.font_manager import FontProperties

def plot_value(data,bins_num):
    #data=count_large[:,0]
    #bins_num=5
    hist, bins = np.histogram(data, bins=bins_num,)
    tot=sum(hist)
    for i in range(bins_num):
        plt.text(bins[i], hist[i] + 2, str(round(float(hist[i])/float(tot),4)*100) + '%', color=sns.xkcd_rgb['nice blue'])


def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'



# 激活函数分布统计
test_result=np.load(r'.\data\frame_score_test_v822.npy')

#sigmoid激活
score1=[]
for i in range(np.shape(test_result)[0]):
    score1.append(round(1/(1+math.e**(-float(test_result[i,2]))),3))

bin_num=5
sns.distplot(score1, kde=False, bins=bin_num, rug=False)
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.xlabel(u'经过激活函数后的评分',fontproperties=font)
plt.ylabel(u'落入x区间的帧个数',fontproperties=font)

plot_value(score1,bin_num)



#改进sigmoid激活
score2=[]
for i in range(np.shape(test_result)[0]):
    score2.append(round(1/(1+math.e**(-float(test_result[i,2])/math.e**4)),3))

bin_num=5
sns.distplot(score2, kde=False, bins=bin_num, rug=False)
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.xlabel(u'经过激活函数后的评分',fontproperties=font)
plt.ylabel(u'落入x区间的帧个数',fontproperties=font)
plot_value(score2,bin_num)


