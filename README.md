# Containers_text_QAN_identification

Data_perpare:<br>
计算箱号的倾斜角度：
1_1angel_detail和1_2combine_angle是用来计算帧的倾斜角度和简单统计，每一帧的角度存在angle_minframe_group.npy中

删除重复较大的帧：
2_1_1IoU为计算每一帧与其上一帧标定框之间的iou，结果存在all_frame_mark_IOU_min.npy中
2_1_2del_highIOU_frame删除iou大的帧，得到的结果为frame_use_all

删除有错误标定的标定框：
2_2_0detect_wrong_loc为依据单个标定与整体标定框的iou，判断哪些是错误的标定并删除。结果为correct_all_use

旋转增强：
2_3_1horizontal_rotation 找出来水平的序列，随机抽角度好的进行旋转，所有的存在all_use_rotation中
2_3_2small_sample_rotation 为序列中帧数在[4,8)区间内的序列，进行旋转补充，存在all_use_add_rotation

不同箱号结构的调整，让不同格式的箱号的分布比较均匀些：
2_2_2 text_structure_stat 得到每个序列的箱号结构，结果为group_structure
2_3_3structure_add 补充不均衡的箱号箱号结构序列。结果all_11为水平1行、all_12为水平2行、all_13为水平3行、all_21为竖直1列、all_22为竖直2列
2_4_3 IoU_cut_2rd 为计算第一行切割的范围和第二行的重合IoU，重合太大的后续会删除。结果为two_line_iou

在每一个序列内根据角度分层抽样：
2_4_4sample_structure 序列内根据角度分层抽样，每个序列抽样3组。抽样的帧组合和标定坐标存在all_choose_v822

拼接训练样本：
3_2patch_join 先把8帧拼起来，再随机加模糊，再随机加漏字符（鉴于训练集中加不加漏字符的情况，对于最终结果印象不大，可以不用），再随机拿一张替换为非箱号。

生成训练数据集的caffe训练读取文件：
4_1_1generate_trian_file_info

生成训练数据集的caffe测试读取文件：
4_2_1generate_train_test_file 将训练数据集的数据做成测试的格式，同时生成测试的caffe读取文件，用于看训练数据集的结果

将训练集的评分结果和其对应的帧一起拼接出来：
5_1train_test_all_score_to_pic

=============================================以上训练步骤完成，下面为验证部分============================================
抽样测试数据集：
6_0_1test_structure_stat测试数据集的基本统计。测试数据集中每一标定框的角度存在6_0_1stat
6_0_2gen_test_file 是采样验证数据集，验证数据集信息存在all_use_frame_test中，caffe文件存在caffe_test_add_incomplete中

验证结果统计：
6_1result_simoid 讨论验证集结果sigmoid的情况
6_2draw_test_result 将网络打的分数，一同人工评分，一起和测试数据集拼接起来
6_3result_stat 结果分析，包括每种箱号结构的分数统计、每种箱号结构的人工分数统计、整体阈值选择讨论

非箱号验证数据集结果统计：
7_1no_container_test

