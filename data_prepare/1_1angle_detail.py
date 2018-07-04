import numpy as np
import math

def cal_angle(coor,way):
    # coor is x1,y1,x2,y2,x3,y3,x4,y4
    # way is '水平' or '垂直'
    #coor=all_recog[1][4:12]
    if way=='水平':
        angle=math.degrees(math.atan(abs(float(coor[1])-float(coor[3]))/abs(float(coor[0])-float(coor[2]))))
        return str(int(angle))
    #coor=all_recog[7][4:12]
    else:
        angle=math.degrees(math.atan(abs(float(coor[0])-float(coor[6]))/abs(float(coor[1])-float(coor[7]))))
        if angle>45:
            return 90-int(angle)
        else:
            return int(angle)



#read data
#filenumber=8
#filename='16_20180530'
def get_angle(filename,filenumber):
    for n in range(1, filenumber + 1):
        #n=1
        f = open(r".\data\%s_%s.txt" % (filename, n))
        # f = open("16_20180530_%s.txt"%n)
        all_raw = []
        line = f.readline().replace('\xef', '').replace('\xbb', '').replace('\xbf', '').replace('\n', '').replace('  ', ' ')
        while line:
            a = line.split(' ')
            all_raw.append(a)
            line = f.readline().replace('\xef', '').replace('\xbb', '').replace('\xbf', '').replace('\n', '').replace('  ',
                                                                                                                      ' ')
        f.close()

        # pic have tag
        all_recog = []
        for i in range(len(all_raw)):
            if all_raw[i][1] != '0':
                all_recog.append(all_raw[i])

        # get coordinates
        all_angle=[]
        for i in range(np.shape(all_recog)[0]):
            if len(all_recog[i])>2:
                if all_recog[i-1][1]=='1':
                    coor_num=(len(all_recog[i])-5)/8
                    coor=[]
                    for j in range(coor_num):
                        coor.append(all_recog[i][4+8*j:4+8*(j+1)])
                    for j in range(coor_num):
                        angle=cal_angle(coor[j],all_recog[i][1])
                        coor[j].append(angle)
                        coor[j].append(all_recog[i][2])
                        coor[j].append(all_recog[i-1][0])
                        all_angle.append(coor[j])

        all_angle=np.array(all_angle)

        np.save(r'.\data\%s_%s_angle'%(filename,n),all_angle)

get_angle('16_20180530',8)
get_angle('17_20180606',10)
get_angle('18_20180608',19)
