import numpy as np
import re
import sys
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf-8')
#read all data
stat=np.load(r'.\data\16_20180530_1_angle.npy')

for i in range(2,9):
    statone=np.load(r'.\data\16_20180530_%s_angle.npy'%i)
    stat=np.vstack((stat,statone))

for i in range(1,11):
    statone=np.load(r'.\data\17_20180606_%s_angle.npy'%i)
    stat=np.vstack((stat,statone))
for i in range(1,20):
    statone=np.load(r'.\data\18_20180608_%s_angle.npy'%i)
    stat=np.vstack((stat,statone))

statpd=pd.DataFrame(stat)
statpd.columns = ['x1', 'y1', 'x2', 'y2', 'x3','y3','x4', 'y4', 'angle', 'id', 'pic']
# 取每个帧所有框的角度均值
grouped_stat = statpd['angle'].astype(int).groupby([statpd[ 'id'],statpd[ 'pic']]).mean()

#grouped_stat.loc['EISU188893542G1']['EmbeddedNetDVR_138.100.103.30_5_20180505094119_20180505100455_1525660392320.mp4_22968.jpg']
statlt=stat.tolist()
avg_angle=[]
for i in range(np.shape(statlt)[0]):
    if statlt[i][9:] not in avg_angle:
        avg_angle.append(statlt[i][9:])

for i in range(np.shape(avg_angle)[0]):
    avg_angle[i].append(grouped_stat.loc[avg_angle[i][0]][avg_angle[i][1]])

# 单独提取出来帧和mp4名
for i in range(np.shape(avg_angle)[0]):
    avg_angle[i].insert(2,int(re.findall(r'mp?4_(.+?).jpg',avg_angle[i][1])[0]))
    avg_angle[i].insert(2, re.findall(r'(.+?).mp?4_', avg_angle[i][1])[0])

###此处需要定义一组视频的划分方式# 一组视频中最小的帧
avg_angle.sort(key=lambda l:(l[0],l[2],l[3]))
avg_angle[0].append(avg_angle[0][3]) #第一个的最小帧
for i in range(1,np.shape(avg_angle)[0]):
    if avg_angle[i][0]==avg_angle[i-1][0] and avg_angle[i][2]==avg_angle[i-1][2] and avg_angle[i][3]-avg_angle[i-1][3]<100: #暂定100
        avg_angle[i].append(avg_angle[i-1][5])
    else:
        avg_angle[i].append(avg_angle[i][3])
#avg_angle为分组信息
np.save(r'.\data\angle_minframe_group',avg_angle)

# 计数一个MP4一个集装箱中，角度大于x的帧数
angle_minframepd=pd.DataFrame(avg_angle)
angle_minframepd.columns = [ 'id', 'pic','mp', 'frame', 'angle','min_frame']
angle_minframepd_sizeallpd = angle_minframepd.groupby([angle_minframepd[ 'id'],angle_minframepd[ 'mp'],angle_minframepd[ 'min_frame']]).size()

angle_minframepd['angle'] = pd.to_numeric(angle_minframepd['angle'], errors='coerce')
angle_minframepd_size9pd=angle_minframepd[angle_minframepd['angle']>=9].groupby([angle_minframepd[ 'id'],angle_minframepd[ 'mp'],angle_minframepd[ 'min_frame']]).size()
angle_minframepd_size14pd=angle_minframepd[angle_minframepd['angle']>=14].groupby([angle_minframepd[ 'id'],angle_minframepd[ 'mp'],angle_minframepd[ 'min_frame']]).size()

#合并
out=np.array(avg_angle).copy().tolist()
for i in range(np.shape(out)[0]):
    out[i].pop(4)
    out[i].pop(3)
    out[i].pop(1)

out_list=[]
for i in range(np.shape(out)[0]):
    if out[i] not in out_list:
        out_list.append(out[i])

#将angle的计数匹配
for i in range(np.shape(out_list)[0]):
    out_list[i].append(angle_minframepd_sizeallpd.loc[out_list[i][0]][out_list[i][1]][int(out_list[i][2])])
    try :
        num9=angle_minframepd_size9pd.loc[out_list[i][0]][out_list[i][1]][int(out_list[i][2])]
    except:
        num9=0
    out_list[i].append(num9)
    try :
        num14=angle_minframepd_size14pd.loc[out_list[i][0]][out_list[i][1]][int(out_list[i][2])]
    except:
        num14=0
    out_list[i].append(num14)

f = open(r'.\data\angle_stat_key_9_14.txt','w')
for one in out_list:
    f.write("%s %s %s %d %d %d\n"%(one[0],one[1],one[2],one[3],one[4],one[5]))
f.close()

np.save(r'.\data\angle_minframe_stat',out_list)

'''
key_value=[]
for i in range(np.shape(stat)[0]):
    key=re.findall(r'(.+?).mp?4_', stat[i,10])[0]+'___'+stat[i,9]
    key_value.append(key)
key_value=np.array(key_value)
key_value=key_value.reshape(len(key_value),1)
stat_key=np.hstack((stat,key_value))
# all frame
angle_list=list(set(stat[:,8]))
angle_stat=[]
for i in range(len(angle_list)):
    count=0
    one_angle_stat=[]
    for j in range(len(stat)):
        if angle_list[i]==stat[j,8]:
            count+=1
    one_angle_stat.append(angle_list[i])
    one_angle_stat.append(count)
    angle_stat.append(one_angle_stat)
angle_stat=np.array(angle_stat)
np.savetxt(r'.\data\angle_stat.txt',angle_stat, fmt='%s',delimiter='\t')
#for every container,>=n's frame percentage
n=9
bad_frame_containers=[]
for i in range(np.shape(stat_key)[0]):
    if int(stat_key[i,8])>=n:
        bad_frame_containers.append(stat_key[i,11])
bad_frame_container=list(set(bad_frame_containers))
all_count=[]
for i in range(len(bad_frame_container)):
    one=[bad_frame_container[i],stat_key[:,11].tolist().count(bad_frame_container[i]),bad_frame_containers.count(bad_frame_container[i])]
    one[0] = one[0].decode('utf8').encode('gbk')
    all_count.append(one)
f = open(r'.\data\angle_stat_key_bgt%d.txt'%n,'w')
for count in all_count:
    f.write("%s %d %d\n"%(count[0],count[1],count[2]))
f.close()
#np.savetxt(r'.\data\angle_stat_key_bgt%d.txt'%n,all_count,fmt='%s',delimiter=',')
'''
for i in range(np.shape(avg_angle)[0]):
    if avg_angle[i][0]=='EISU188893542G1':#and avg_angle[i][2]=='EmbeddedNetDVR_138.100.103.30_5_20180505094119_20180505100455_1525660392320':
        print(avg_angle[i])
