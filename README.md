# Containers_text_QAN_identification

Data_perpare:


1_1angle_detail:计算标记文本框的倾斜角度，区分水平和垂直标记。
1_2combine_angle:一帧图对应多个标记框，图的角度为图中标记框的角度平均。
2_1IOU: 计算每一个框与其上一个连续帧中对应框之间的iou。
2_2del_highIOU_frame:删除总帧数小于4和iou大于0.9的帧
2_3_1random_sample: 纯随机采样
2_3_2random_sample_angle: 根据倾斜角度大于等于9和小于9作为分层采样的标准，
3_1patch_join:一组序列拼接为一个图像
4_1generate_file_info: 为caffe训练制作文本信息

