import numpy as np
import re
import cv2
import math
import os

'''
poo12 和 pool3有后缀
没有后缀的为pool4
'''

def write_score(s,color=(255,255,255)):
    font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
    im=np.zeros((64,100,3),np.uint8)
    img=cv2.putText(im,s,(0,20),font,0.43,color,1)#添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
    #imshow(img)frame_score[0,3]
    return img

def cal_contast(image):
    # 彩图
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    sum_all=0
    count=0
    for i in range(np.shape(image_gray)[0]):
        for j in range(np.shape(image_gray)[1]):
            this=0
            try:
                this=this+(image_gray[i,j]-image_gray[i-1,j])**2
                count+=1
            except:
                count -= 1
            try:
                this = this + (image_gray[i, j] - image_gray[i + 1, j]) ** 2
                count += 1
            except:
                count -= 1
            try:
                this = this + (image_gray[i, j] - image_gray[i, j-1]) ** 2
                count += 1
            except:
                count -= 1
            try:
                this = this + (image_gray[i, j] - image_gray[i, j+1]) ** 2
                count += 1
            except:
                count -= 1
            sum_all=sum_all+this
    contract=sum_all/count
    return contract

#test图像原始信息
all_use_frame=np.load(r'.\data\all_use_frame_test.npy')

f=open(r'.\data\rengong_score.txt','r')
line=f.readline().replace('\n','')
other_socre=[]
while line:
    other_socre.append(line.split('\t'))
    line = f.readline().replace('\n','')
f.close()
other_socre=np.array(other_socre)

#test图像caffe评分
test_result=np.load(r'.\data\frame_score_test_incomplete_v822_trainwithoutincomplete.npy')
group=list(set(test_result[:,0]))


#i=120
test_result_all=[]
for i in range(np.shape(group)[0]):
    this_group=test_result[np.where(test_result[:,0]==group[i])]

    # 组内所有帧的详细信息
    this_group_loc=np.zeros([1,14])
    # contrast=[]
    #bright=[]
    for j in range(np.shape(this_group)[0]):
        pic=re.findall('^(?:add)?(.*?)$',this_group[j,1].split('_')[0])[0]
        for a in this_group[j, 0].split('_')[2:-1]:
            pic=pic+'_'+a
        pic=pic+'.'+re.findall('(mp*4_[0-9]+)',this_group[j,1])[0]+'.'+'jpg'
        this_group_loc=np.vstack((this_group_loc,all_use_frame[np.where(all_use_frame[:,10]==pic)]))
        #image=cv2.imread(r'E:\test_cut\%s\%s\%s\%s'\
        #                 %(this_group[j,0].split('_')[0],re.findall('(.*?).mp*4',pic)[0],this_group[j, 0].split('_')[-1],pic))
        #bright.append(np.mean(image))
        #contrast.append(cal_contast(image))
    this_group_loc=this_group_loc[1:]
    max_socre=max([round(1/(1+math.exp(-float(s)/(math.e**4))),4) for s in this_group[:,2]]) # TODO 如果sigmoid改了这里要改

    # 角度
    percentile25=np.percentile([int(n) for n in this_group_loc[:,8]],25)
    percentile75=np.percentile([int(n) for n in this_group_loc[:,8]],75)
    #contrast25=np.percentile(contrast,25)
    #bright25=np.percentile(bright,25)

    image_show=np.zeros([1,492,3],np.uint8)
    for j in range(np.shape(this_group)[0]):
        #pic=re.findall('^(?:add)?(.*?)$',this_group[j,1].split('_')[0])[0]
        pic=this_group[j,1].split('_')[0]
        for a in this_group[j, 0].split('_')[2:-1]:
            pic=pic+'_'+a
        pic=pic+'.'+re.findall('(mp*4_[0-9]+)',this_group[j,1])[0]+'.'+'jpg'
        image=cv2.imread(r'..\test_cut_add_incomplete\%s\%s\%s\%s'\
                         %(this_group[j,0].split('_')[0],re.findall('(?:add)?(.*?).mp*4',pic)[0],this_group[j, 0].split('_')[-1],pic))
        '''
        cv2.imshow("image", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        '''
        # 角度
        this_pic_loc=all_use_frame[np.where(all_use_frame[:,10]==re.findall('^(?:add)?(.*?)$',pic)[0])]

        angle=int(np.mean([int(a) for a in this_pic_loc[:,8]]))
        if angle<=percentile25 or angle<=2:
            angle_score=1
        elif (angle>percentile25 and angle<=percentile75) or (angle>2 and angle<9):
            angle_score=0.5
        else:
            angle_score=0
        #if contrast[j]<contrast25:
        #    contrast_score=0
        #else:
        #    contrast_score=1
        #if bright[j]<bright25:
        #    bright_score=0
        #else:
        #    bright_score=1
        if j<=7:
            other=other_socre[np.where(other_socre[:,0]==pic),1][0][0]
        else:
            other=0

        image_angle=write_score('angle: '+str(angle_score)+'('+str(angle)+')')
        this_image_show=np.hstack((image,image_angle))
        #image_contrast=write_score('cont_socre: '+str(contrast_score))
        #this_image_show=np.hstack((this_image_show,image_contrast))
        #image_bright=write_score('bright_socre: '+str(bright_score))
        image_other=write_score('other: '+str(other))
        this_image_show=np.hstack((this_image_show,image_other))


        # 打分
        score=round(1/(1+math.exp(-float(this_group[j,2])/(math.e**4))),4) # TODO 如果sigmoid改了这里要改
        if score==max_socre:
            image_score=write_score('score: '+str(score),color=(0,255,0))
        else:
            image_score=write_score('score: '+str(score))
        this_image_show=np.hstack((this_image_show,image_score))

        image_show=np.vstack((image_show,this_image_show))
        test_result_all.append([this_group[j,0].split('_')[0]+'_'+re.findall('(.*?).mp*4',pic)[0]+'_'+this_group[j, 0].split('_')[-1],this_group_loc[0,9],this_group_loc[0,10],this_group_loc[0,13],this_group_loc[0,11],angle,angle_score,other,score,max_socre])

        '''
        cv2.imshow("image_show", image_show)
        cv2.waitKey()
        cv2.destroyAllWindows()
        '''
    if not os.path.exists(r'..\frame_score_test_incomplete_v822_trainwithoutincomplete'):
        os.mkdir(r'..\frame_score_test_incomplete_v822_trainwithoutincomplete')
    cv2.imwrite(r'..\frame_score_test_incomplete_v822_trainwithoutincomplete\%s.jpg'%(this_group[j,0].split('_')[0]+'_'+re.findall('(.*?).mp*4',pic)[0]+'_'+this_group[j, 0].split('_')[-1]),image_show)

np.save(r'.\data\frame_score_test_incomplete_v822_trainwithoutincomplete',test_result_all)




