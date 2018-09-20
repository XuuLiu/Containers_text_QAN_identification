import cv2
import re
import numpy as np
import os
#训练集的情况

f=open(r'.\data\caffe_file_list_train_v822.txt','r')
file_lines = f.readlines()
f.close()

caffe_test_file_list=[]

for N in range(119): #np.shape(file_lines)[0]
    n=N*50
    file_name=re.findall("/data/liuxu/container_qan/train_data/(.*?) ",file_lines[n])[0]
    img=cv2.imread(r'..\patch_frame_incomplete_v822\%s'%file_name)


    for i in range(8):
        one_frame=img[64*i:64*(i+1)]
        feat_save_dir = r'..\patch_frame_test_incomplete_v822\%s\%s' % (n, re.findall('_([0-9]*?).jpg',file_name)[0])
        if not os.path.exists(feat_save_dir):
            os.makedirs(feat_save_dir)
        cv2.imwrite(r'..\patch_frame_test_incomplete_v822\%s\%s\%s.jpg' % (n, re.findall('_([0-9]*?).jpg',file_name)[0],re.findall('_([0-9]*?).jpg',file_name)[0]+'_'+str(i)), one_frame)
        caffe_test_file_list.append(r'/data/liuxu/container_qan/train_test_data_v822/%s/%s/%s.jpg' % (n, re.findall('_([0-9]*?).jpg',file_name)[0],re.findall('_([0-9]*?).jpg',file_name)[0]+'_'+str(i)))



fl=open(r'.\data\caffe_file_list_traintest_incomplete_v822.txt','w')
for one in caffe_test_file_list:
    fl.write('%s 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 38\n'%one)
f.close()


# cv2.imshow("one",one_frame)
# cv2.waitKey()#
# cv2.destroyAllWindows()