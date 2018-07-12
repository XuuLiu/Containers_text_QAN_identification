import numpy as np

# 为caffe训练建立文档

def encoder(str):
    decoder=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U','V',
            'W', 'X', 'Y', 'Z']

    str_encoder=[]
    for i in range(len(str)):
        str_encoder.append(decoder.index(str[i].upper()))

    for i in range(20-len(str)):
        str_encoder.append(38)
    return str_encoder


all_choose=np.load(r'.\data\all_choose2.npy').tolist()
# all_choose[组index][组内采样index][0:8样本小组内的帧index][帧的标记框index][0:13属性] #组内多次采样
# all_choose[组index]               [0:8样本小组内的帧index][帧的标记框index][0:13属性] #组内单次采样

file_list=[] #拼接好的文件名，和
for n in range(np.shape(all_choose)[0]):
    if len(np.shape(all_choose[n]))<=3 and np.shape(all_choose[n])[0]==8: #1次采样
        file_name='%s_%s_%s.jpg'%(all_choose[n][-1][0][9],all_choose[n][-1][0][11],all_choose[n][-1][0][12])
        file_label=all_choose[n][-1][0][9]
        file_list.append([file_name,file_label])
    else:
        for i in range(np.shape(all_choose[n])[0]):
            file_name = '%s_%s_%s.jpg' % (all_choose[n][i][-1][0][9], all_choose[n][i][-1][0][11], all_choose[n][i][-1][0][12])
            file_label = all_choose[n][i][-1][0][9]
            file_list.append([file_name, file_label])


file_list_caffe=[]
for n in range(np.shape(file_list)[0]):
    enc=encoder(file_list[n][1])
    enc.insert(0,file_list[n][0])
    file_list_caffe.append(enc)



f=open(r'.\data\caffe_file_list_train2.txt','w')
for one in file_list_caffe:
    f.write('/data/liuxu/container_qan/train_data/%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n'\
            %(one[0],one[1],one[2],one[3],one[4],one[5],one[6],one[7],one[8],one[9],one[10],one[11],one[12],one[13],one[14],one[15],one[16],one[17],one[18],one[19],one[20]))
f.close()


