import cv2
import numpy as np
from math import *
import random as rdm
import math
'''
增加漏字符的训练样本
'''

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

# 运动模糊
def genaratePsf(length, angle):
    EPS = np.finfo(float).eps
    alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1;
    # 模糊核大小
    sx = int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS))
    sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
    psf1 = np.zeros((sy, sx))

    # psf1是左上角的权值较大，越往右下角权值越小的核。
    # 这时运动像是从右下角到左上角移动
    for i in range(0, sy):
        for j in range(0, sx):
            psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
            rad = math.sqrt(i * i + j * j)
            half = length / 2
            if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp * temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j]);
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    # 运动方向是往左上运动，锚点在（0，0）
    anchor = (0, 0)
    # 运动方向是往右上角移动，锚点一个在右上角
    # 同时，左右翻转核函数，使得越靠近锚点，权值越大
    if angle < 90 and angle > 0:
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, 0)
    elif angle > -90 and angle < 0:  # 同理：往右下角移动
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)

    elif anchor < -90:  # 同理：往左下角移动
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)
    psf1 = psf1 / psf1.sum()

    return psf1, anchor

# kernel,anchor=genaratePsf(10,5)
# motion_blur=cv2.filter2D(cut_res,-1,kernel,anchor=anchor)
'''
cv2.startWindowThread()
cv2.imshow("motion_blur", motion_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

def get_patch_frame(frame_group):
    # frame_group 由小于等于8个帧组成，(帧数，帧内标记框个数，14个属性)，属性分别为['x1','y1','x2','y2','x3','y3','x4','y4','angle', 'id', 'pic','mp', 'frame','min_frame']
    '''
    frame_group=all_choose[1000]

    change_index 为随机替换为非箱号的index
    is_motion_blur 决定该帧是否运动模糊，[1,0,0,0,0,0,0,0] 随机，1/8概率
    is_gaussian_blur 决定该帧是否高斯模糊，[1,0,0,0,0,0,0,0] 随机，1/8概率
    is_up_cut 决定上部是否剪切为漏检，[1,0,0,0,0,0,0,0] 随机，1/8概率
    is_down_cut 决定下部是否剪切为漏检，[1,0,0,0,0,0,0,0] 随机，1/8概率
    is_left_cut 决定左部是否剪切为漏检，[1,0,0,0,0,0,0,0] 随机，1/8概率
    is_right_cut 决定下部是否剪切为漏检，[1,0,0,0,0,0,0,0] 随机，1/8概率
    '''
    patch_frame=np.zeros((1,192,3),np.uint8)
    change_index = np.floor(np.random.random(1)[0] * 7) #最后一个因为涉及到图片的名字，故不用不是箱号的图
    is_motion_blur=[1,0,0,0,0,0,0,0]
    is_gaussian_blur = [1, 0, 0, 0, 0,0,0,0]
    is_up_cut=[1, 0, 0, 0, 0,0,0,0] #如果不想要缺字符，这里全改为0即可
    is_down_cut=[1, 0, 0, 0, 0,0,0,0] #如果不想要缺字符，这里全改为0即可
    is_left_cut=[1, 0, 0, 0, 0,0,0,0] #如果不想要缺字符，这里全改为0即可
    is_right_cut=[1, 0, 0, 0, 0,0,0,0] #如果不想要缺字符，这里全改为0即可

    for j in range(8):
        if j <np.shape(frame_group)[0] and j!=change_index: #集装箱的图
            detail=frame_group[j] #一张图的全部标记框

            image = cv2.imread(r'..\frame_rotation\%s' % detail[0][10], cv2.IMREAD_COLOR)
            try:
                if image == None:
                    image = cv2.imread(r'..\all_frame\%s' % detail[0][10], cv2.IMREAD_COLOR)
                else:
                    pass
            except:
                pass

            '''
            cv2.imshow("Original", image)
            cv2.waitKey()
            cv2.destroyAllWindows()
            '''

            x_list=[]
            y_list=[]
            for i in range(np.shape(detail)[0]):
                x_list.append(int(detail[i][0]))
                x_list.append(int(detail[i][2]))
                x_list.append(int(detail[i][4]))
                x_list.append(int(detail[i][6]))
                y_list.append(int(detail[i][1]))
                y_list.append(int(detail[i][3]))
                y_list.append(int(detail[i][5]))
                y_list.append(int(detail[i][7]))


            min_x=max(min(x_list)-10,0)
            min_y=max(min(y_list)-10,0)
            max_x=min(max(x_list)+10,image.shape[1])
            max_y=min(max(y_list)+10,image.shape[0])

            cut = image[min_y:max_y,min_x:max_x]


            height, width = cut.shape[:2]
            if float(height)/float(width)>1.1: #改这里
                cut_rotation=rotation(cut,degree=90)
                cut_res=cv2.resize(cut_rotation,(192,64))
            else:
                cut_res = cv2.resize(cut, (192, 64))
            # 在这拼接好的8帧中，随机添加模糊
            # 运动模糊
            if rdm.sample(is_motion_blur,1)[0]==1:
                kernel,anchor=genaratePsf(10,5)
                motion_blur=cv2.filter2D(cut_res,-1,kernel,anchor=anchor)
                print('m'+str(j))
            else:
                motion_blur=cut_res

            # 高斯模糊
            if rdm.sample(is_gaussian_blur, 1)[0] == 1:
                gaussian_blur = cv2.GaussianBlur(motion_blur, (5,5), 1.5)
                print('g'+str(j))
            else:
                gaussian_blur = motion_blur
            #随机增加漏字符
            #上漏检
            if rdm.sample(is_up_cut, 1)[0] == 1:
                up_cut = gaussian_blur[13:]
                print('up_cut'+str(j))
            else:
                up_cut = gaussian_blur

            #下漏检
            if rdm.sample(is_down_cut, 1)[0] == 1:
                down_cut = up_cut[:51]
                print('down_cut'+str(j))
            else:
                down_cut = up_cut

            #左漏检
            if rdm.sample(is_left_cut, 1)[0] == 1:
                left_cut = down_cut[:,38:]
                print('left_cut'+str(j))
            else:
                left_cut = down_cut

            #右漏检
            if rdm.sample(is_right_cut, 1)[0] == 1:
                right_cut = left_cut[:,:154]
                print('right_cut'+str(j))
            else:
                right_cut = left_cut

            cut_res2=cv2.resize(right_cut,(192,64))



            '''
             cv2.imshow("Original", cut_res)
             cv2.waitKey()
             cv2.destroyAllWindows()
             '''

            patch_frame=np.vstack((patch_frame,cut_res2))
        else:# 非集装箱的文字
            this_no_number=rdm.sample(add_no_number,1)
            image = cv2.imread(r'..\all_frame\%s' % this_no_number[0][0], cv2.IMREAD_COLOR)
            try:
                if image == None:
                    image = cv2.imread(r'..\all_frame_test\%s' % this_no_number[0][0], cv2.IMREAD_COLOR)
                else:
                    pass
            except:
                pass
            min_x=int(this_no_number[0][1])
            min_y=int(this_no_number[0][2])
            max_x=int(this_no_number[0][3])
            max_y=int(this_no_number[0][4])

            cut = image[min_y:max_y,min_x:max_x]
            height, width = cut.shape[:2]
            if float(height)/float(width)>1.1: #改这里
                cut_rotation=rotation(cut,degree=90)
                cut_res=cv2.resize(cut_rotation,(192,64))
            else:
                cut_res = cv2.resize(cut, (192, 64))
            '''
             cv2.imshow("Original", cut_res)
             cv2.waitKey()
             cv2.destroyAllWindows()
             '''
            patch_frame=np.vstack((patch_frame,cut_res))
    patch_frame=np.delete(patch_frame,0,axis=0)
    '''
    cv2.imshow("patch_frame", patch_frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    #训练集的样本存在这里
    cv2.imwrite(r'..\patch_frame_incomplete_v822\%s_%s_%s.jpg' % (detail[0][9], detail[0][11], detail[0][12]), patch_frame) #命名规则：箱号_视频号_采样中的最后一个帧的帧号
    return 0



all_choose=np.load(r'.\data\all_choose_v822.npy').tolist()
# [序列index][:7序列内帧index][帧内标定框index][]

# 每组采样中，都加一个不是箱号的图
add_no_number=loadDataSet(r'.\data\add_no_number_train.txt')
#区分组内采样的个数

#n=5500

for n in range(np.shape(all_choose)[0]):
    try:
        get_patch_frame(all_choose[n])
    except:
        pass
