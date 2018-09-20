import numpy as np
#训练集caffe
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


all_choose=np.load(r'.\data\all_choose_v822.npy').tolist()
# [序列index][:7序列内帧index][帧内标定框index][]

file_list=[] #拼接好的文件名
for n in range(np.shape(all_choose)[0]):
    try:
        file_name='%s_%s_%s.jpg'%(all_choose[n][-1][0][9],all_choose[n][-1][0][11],all_choose[n][-1][0][12])
        file_label=all_choose[n][-1][0][9]
        file_list.append([file_name,file_label])
    except:
        pass


file_list_caffe=[]
for n in range(np.shape(file_list)[0]):
    enc=encoder(file_list[n][1])
    enc.insert(0,file_list[n][0])
    file_list_caffe.append(enc)



f=open(r'.\data\caffe_file_list_train_v822.txt','w')
for one in file_list_caffe:
    f.write('/data/liuxu/container_qan/train_data/%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n'\
            %(one[0],one[1],one[2],one[3],one[4],one[5],one[6],one[7],one[8],one[9],one[10],one[11],one[12],one[13],one[14],one[15],one[16],one[17],one[18],one[19],one[20]))
f.close()


