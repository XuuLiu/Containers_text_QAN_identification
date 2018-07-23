# Containers_text_QAN_identification

Data_perpare:<br>
1_1angle_detail:计算标记文本框的倾斜角度，区分水平和垂直标记。<br>
1_2combine_angle:一帧图对应多个标记框，图的角度为图中标记框的角度平均。<br>
2_1IOU: 计算每一个框与其上一个连续帧中对应框之间的iou。<br>
2_2del_highIOU_frame:删除总帧数小于4和iou大于0.9的帧<br>
2_3_1random_sample: 纯随机采样<br>
2_3_2random_sample_angle: 根据倾斜角度大于等于9和小于9作为分层采样的标准，<br>
3_1patch_join:一组序列拼接为一个图像<br>
4_1generate_file_info: 为caffe训练制作文本信息<br>
5_1read_test_file: 读取测试数据集，并筛选抽样<br>
5_2test_result:测试结果拼接<br>
6_1result_stat:测试结果统计<br>
