import cv2
import numpy as np
import math
import random as rdm
import re

#非集装箱号文本测试的样本生成、结果解析

def loadDataSet(filename):   #读取文件，txt文档
    # filename=r'.\data\add_no_number_train.txt'
    fr = open(filename,'r')
    dataMat = []
    line=fr.readline().replace('\n','')
    while line:
        a = line.split(' ')
        dataMat.append(a)
        line = fr.readline().replace('\n','')
    out_data=[]
    for i in range(np.shape(dataMat)[0]):
        if len(dataMat[i])==1:
            this_line=[]
            this_line.append(dataMat[i][0])
        else:
            this_line.append(dataMat[i][0])
            this_line.append(dataMat[i][1])
            this_line.append(dataMat[i][2])
            this_line.append(dataMat[i][3])
            out_data.append(this_line)
    return out_data

def rotation(img,degree):
    height, width = img.shape[:2]
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    #旋转中心
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation


#非箱号样本生成
add_no_number=loadDataSet(r'.\data\add_no_number_test.txt')
f=open(r'.\data\caffe_test_add_no_container.txt','w')
for i in range(np.shape(add_no_number)[0]):
    image=cv2.imread(r'..\all_frame_test\%s.jpg'%(add_no_number[i][0]))
    try:
        if image == None:
            image = cv2.imread(r'..\all_frame\%s.jpg' % add_no_number[i][0], cv2.IMREAD_COLOR)
        else:
            pass
    except:
        pass

    min_x = int(add_no_number[i][1])
    min_y = int(add_no_number[i][2])
    max_x = int(add_no_number[i][3])
    max_y = int(add_no_number[i][4])

    cut = image[min_y:max_y, min_x:max_x]
    height, width = cut.shape[:2]
    if float(height) / float(width) > 1.5:  # 改这里
        cut_rotation = rotation(cut, degree=90)
        cut_res = cv2.resize(cut_rotation, (192, 64))
    else:
        cut_res = cv2.resize(cut, (192, 64))
    '''
    cv2.imshow("cut_res", cut_res)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    cv2.imwrite(r'E:\test_cut\OTHER\0\0\%s.jpg'%(i),cut_res)
    f.write('/data/liuxu/container_qan/test_data/OTHER/0/0/%s.jpg 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 38\n'%i)

f.close()


####################非箱号结果拼图

def write_score(s,color=(255,255,255)):
    font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
    im=np.zeros((64,100,3),np.uint8)
    img=cv2.putText(im,s,(0,20),font,0.4,color,1)#添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
    #imshow(img)frame_score[0,3]
    return img

#拼图
test_no_result=np.load(r'.\data\frame_score_test_nocontianer_v822.npy')
add_no_number=loadDataSet(r'.\data\add_no_number_test.txt')

show_image=np.zeros([1,292,3],np.uint8)
for i in range(np.shape(test_no_result)[0]):
    test_no_result[i,2]=round(1/(1+math.exp(-float(test_no_result[i,2])/(math.e**4))),2) #TODO 改sigmoid
    image = cv2.imread(r'..\test_cut\OTHER\0\0\%s.jpg' % (i))
    this_imnage=np.hstack((image,write_score('score: '+str(test_no_result[i,2]))))
    show_image=np.vstack((show_image,this_imnage))
    '''
    cv2.imshow("show_image", show_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
cv2.imwrite(r'..\test_result_pool4_v822\on_container_v822.jpg',show_image)

#############################################看非箱号哪些的比序列内帧最大分大
#图片对应的非箱号评分
test_no_result=np.load(r'.\data\frame_score_test_nocontianer_v822.npy')
add_no_number=loadDataSet(r'.\data\add_no_number_test.txt')
test_result_all=np.load(r'.\data\test_result_all_pool4_v822.npy')

no_score=[]
for i in range(np.shape(add_no_number)[0]):
    no_score.append([add_no_number[i][0],round(1/(1+math.exp(-float(test_no_result[i,2])/(math.e**4))),2)]) #TODO 改sigmoid

count=0
for i in range(np.shape(no_score)[0]):
    this_group_0=test_result_all[np.where(test_result_all[:,4]==re.findall('^(.*?).mp?4_[0-9]+$',no_score[i][0])[0])]
    this_group=[]
    for k in range(np.shape(this_group_0)[0]):
        if int(re.findall('.*?_([0-9]+)$',this_group_0[k,0])[0])<=int(re.findall('.*?_([0-9]+)$',no_score[i][0])[0]):
            this_group.append(this_group_0[k].tolist())
    this_group=np.array(this_group)
    if len(this_group)>0:
        if min([float(k) for k in this_group[:,9]])<no_score[i][1]:
            count+=1
            print(this_group)


max([float(k) for k in test_result_all[:,9]])

###############
# 非箱号评分分布
def plot_percentile(data,bins_num,per):
    hist, bins = np.histogram(data, bins=bins_num,)
    x=np.percentile(data,per)
    for i in range(np.shape(bins)[0]-1):
        if bins[i]<=x and bins[i+1]>=x:
            index=i
    y=hist[index]
    plt.plot([x,x,],[0,y],'k--',linewidth=1.5,c=sns.xkcd_rgb['nice blue'])
    plt.text(x+0.01,0.2,str(round(x,2)),color=sns.xkcd_rgb['nice blue'])
    plt.text(x-(bins[1]-bins[0]),y+0.4,str(per)+'%',color=sns.xkcd_rgb['nice blue'])

test_no_result=np.load(r'.\data\frame_score_test_nocontainer.npy')
for i in range(np.shape(test_no_result)[0]):
    test_no_result[i, 2] = round(1 / (1 + math.exp(-float(test_no_result[i, 2]) / (math.e ** 2))), 2)

no_container_score=[float(score) for score in test_no_result[:,2]]
sns.distplot(no_container_score, kde=False, bins=20, rug=False)
plot_percentile(no_container_score,20,50)
plot_percentile(no_container_score,20,5)
plot_percentile(no_container_score,20,95)

###########################
test_no_result=np.load(r'.\data\frame_score_test_nocontianer_v822.npy')
for i in range(np.shape(test_no_result)[0]):
    test_no_result[i, 2] = round(1 / (1 + math.exp(-float(test_no_result[i, 2]) / (math.e ** 4))), 2)

threthold=0.5
count=0
for i in test_no_result[:,2]:
    if float(i) >=threthold:
        count+=1
print(float(count)/float(np.shape(test_no_result)[0]))