import numpy as np
from pylab import *
import cv2

#sore minmax normalization
frame_score=np.load(r'.\data\frame_score_train_incomplete_v822_trainwithincomplete.npy')

score=[]
for n in range(np.shape(frame_score)[0]):
    min=np.min([float(i) for i in frame_score[(n-n%8):(n-n%8+8),2]])
    max=np.max([float(i) for i in frame_score[(n-n%8):(n-n%8+8),2]])
    score.append(round((float(frame_score[n,2])-min)/(max-min),4))

frame_score=np.hstack((frame_score,np.array(score).reshape([len(score),1])))


# 写字
def write_score(s):
    font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
    im=np.zeros((64,100,3),np.uint8)
    img=cv2.putText(im,s,(0,20),font,0.4,(255,255,255),1)#添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
    #imshow(img)frame_score[0,3]
    return img

# n 组inddex， (n-n%8):(n-n%8+8)

for n in range(100):#np.shape(frame_score)[0]):
    group_img=np.zeros([1,292,3],np.uint8)
    for i in range(n*8,(n+1)*8):
        # i=0 全部index
        img_raw=cv2.imread('..\\patch_frame_test_incomplete_v822\\'+frame_score[i,0].split('_')[0]+'\\'+frame_score[i,0].split('_')[1]+'\\'+frame_score[i,1]+'.jpg')
        img_show=np.hstack((img_raw,write_score(frame_score[i,3][:np.min([6,len(frame_score[i,3])])])))
        group_img=vstack((group_img,img_show))
    group_img = np.delete(group_img, 0, axis=0)
    #imshow(group_img)
    cv2.imwrite(r'..\traintest_result_incomplete_v822\%s.jpg'%n,group_img)
