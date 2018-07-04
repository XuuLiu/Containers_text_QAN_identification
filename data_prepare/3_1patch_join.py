import cv2
import numpy as np
from math import *

#竖直的文本旋转、根据标记框裁剪、8个拼接起来

def rotation(img,degree):
    height, width = img.shape[:2]
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation

def get_patch_frame(frame_group):
    # frame_group 由小于等于8个帧组成，(帧数，帧内标记框个数，14个属性)，属性分别为['x1','y1','x2','y2','x3','y3','x4','y4','angle', 'id', 'pic','mp', 'frame','min_frame']
    patch_frame=np.zeros((1,192,3))
    for j in range(8):
        if j <np.shape(frame_group)[0]:
            detail=frame_group[j] #一张图的全部标记框

            image_path= r'E:\all_frame\%s'%detail[0][10]
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
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
            if height>width:
                cut_rotation=rotation(cut,degree=90)
                cut_res=cv2.resize(cut_rotation,(192,64))
            else:
                cut_res = cv2.resize(cut, (192, 64))
            patch_frame=np.vstack((patch_frame,cut_res))
        else:
            patch_frame = np.vstack((patch_frame, np.random.random((64,192,3))*225))

    patch_frame=np.delete(patch_frame,0,axis=0)
    cv2.imwrite(r'E:\patch_frame\%s_%s_%s.jpg' % (detail[0][9], detail[0][11], detail[0][12]), patch_frame)
    return np.uint8(patch_frame)


all_choose=np.load(r'.\data\all_choose2.npy').tolist()
# all_choose[组index][组内采样index][0:8样本小组内的帧index][帧的标记框index][0:13属性] #组内多次采样
# all_choose[组index]               [0:8样本小组内的帧index][帧的标记框index][0:13属性] #组内单次采样

#区分组内采样的个数
all_patch_frame=[]
for n in range(np.shape(all_choose)[0]):
    patch_frame=[]
    if len(np.shape(all_choose[n]))<=3: #1次采样
        patch_frame.append(get_patch_frame(all_choose[n]))
    else:
        for j in range(np.shape(all_choose[n])[0]): #多次采样
            patch_frame.append(get_patch_frame(all_choose[n][j]))
    all_patch_frame.append(patch_frame)




# cv2.imshow("patch_frame",all_patch_frame[4][0])
# cv2.waitKey()
# cv2.destroyAllWindows()


