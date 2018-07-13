import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import cv2

import sys

caffe_root = '/data/liuxu/Darwin-Caffe-lstm/'
sys.path.append(caffe_root + 'python')

import caffe

import os

# config
caffe.set_mode_gpu()
caffe.set_device(1)

#get solver
solver = caffe.SGDSolver('/data/liuxu/container_qan/prototxt/solver_container_test_pool4.prototxt')
# recog_solver = caffe.SGDSolver(caffe_root+'xinjiang/proto/solver_test_recog_solver.prototxt')
#print('load model:')
#for k, v in solver.test_nets[0].blobs.items():
#    print('blob name : ' + str(k) + ' , its shape : ' + str(v.data.shape))
#pass

# loading model
#solver.test_nets[0].copy_from('/data1/icdar/tmp/caffemodel/lujing/useful_models/xj_chepai_qan_full_iter_300000.caffemodel')
#solver.test_nets[0].copy_from('/data1/icdar/tmp/caffemodel/lujing/xinjiang_qan/xj_chepai_qan_pool4_8_batch1_iter_30000.caffemodel')
solver.test_nets[0].copy_from('/data/liuxu/container_qan/model_3/_iter_30000.caffemodel') # model 

#test file list
#test_file = open('/data1/icdar/xinjiang/ataile_croped_license_test_seq/ataile_croped_license_test_seq_fakelist_8.txt', 'r')
test_file = open('/data/liuxu/container_qan/caffe_file_list_test.txt', 'r')
test_file_lines = test_file.readlines()
test_file.close()

print('a total of ' + str(len(test_file_lines)) + ' test images')

car_num = 0

last_car_index = '1_1'
cur_acc = 1
cur_car_index = '1_1'
cur_score = -1
cur_frame_index = -1

first_car_flag = True

feat_save_dir = './feat_save'
if not os.path.exists(feat_save_dir):
    os.mkdir(feat_save_dir)
save_info=[]

for test_iter in range(0, len(test_file_lines)):
    # get car index
    cur_car_index = test_file_lines[test_iter].split('/')[5] + '_' + test_file_lines[test_iter].split('/')[6] 
    car_frame_index = test_file_lines[test_iter].split('/')[7].split('.')[0]

    if first_car_flag == True:
        last_car_index = cur_car_index
        first_car_flag = False
    pass

    if not cur_car_index == last_car_index:
        car_num = car_num + 1
        print('car ' + str(last_car_index) + ' max score(index ' + str(cur_frame_index) + ') : ' + str(
            cur_score) + '\n')
        last_car_index = cur_car_index
        cur_score = -1
    pass
	
    print('processing car index : ' + cur_car_index + ' ; frame index : ' + str(car_frame_index)),
    solver.test_nets[0].forward()
    
    # recog_solver.test_nets[0].forward()

	#get qualit score
    score = solver.test_nets[0].blobs['reshape_s1'].data[0, 0]  #score_output  #reshape_s1
    print(' score : ' + str(score))
    save_info.append([cur_car_index,car_frame_index,score])
	


    if score > cur_score:
        cur_score = score
        cur_frame_index = car_frame_index
    pass

    if test_iter == (len(test_file_lines) - 1):
        car_num = car_num + 1
        print('car ' + str(cur_car_index) + ' max score(index ' + str(cur_frame_index) + ') : ' + str(
            cur_score) + '\n')
    pass
	

pass

np.save(feat_save_dir+'/frame_score2',save_info)
