#!/bin/bash

#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

cd /data/liuxu/Darwin-Caffe-lstm
#****************************************Config Info*******************************************
#AUTHOR_SERVER_IP="10.17.135.222"
AUTHOR_SERVER_IP="10.65.223.83"

AUTHOR_SERVER_PORT=17192

#darwin=darwin-caffe base=bvlc-caffe
RUN_TYPE="darwin"

#MACHINE_LIST="10.65.223.35"

MACHINE_LIST="10.3.78.24"

PARAMETER_LIST="train -solver=/data/liuxu/container_qan/prototxt/solver_container_pool4.prototxt --weights=/data/liuxu/container_qan/model_2/_iter_30000.caffemodel"

GPU_LIST="2,3"

#**********************************************************************************************


#如果命令行自带参数，则使用命令行的参
while getopts "t:m:p:g:" arg
do
	case $arg in
		t)
			RUN_TYPE=$OPTARG
			;;
		m)
			MACHINE_LIST=$OPTARG
			;;
		p)
			PARAMETER_LIST=$OPTARG
			;;
		g)
			GPU_LIST=$OPTARG
			;;
	esac
done

echo $RUN_TYPE
echo $MACHINE_LIST
echo $PARAMETER_LIST
echo $GPU_LIST



machine_num="0"
gpu_num="0"
process_num="0"
GPU_LIST_=${GPU_LIST//,/ }

echo "" > ./hosts

for gpu in $GPU_LIST_; do
	let "gpu_num=gpu_num+1"
done

for machine in $MACHINE_LIST; do
	echo "$machine:$gpu_num" >> ./hosts
	let "machine_num=machine_num+1"
done

let "process_num=machine_num*gpu_num"



ANS_SGD_ENABLE_VALUE="0"
ANS_SGD_TYPE_VALUE="2"  #1=downpour sgd; 2=ea sgd
ANS_SGD_PUSH_STEP_VALUE="10000000"
ANS_SGD_FETCH_STEP_VALUE="10000000"
ANS_SGD_LEARN_RATE_VALUE="0.001"
ANS_SGD_EA_ROU_value="1.0"



if [ "${RUN_TYPE}" = "darwin" ]; then
	/usr/mvapich_rc1/bin/mpirun \
		-env MV2_USE_CUDA 1 \
		-env MV2_ENABLE_AFFINITY 0  \
		-env ANS_SGD_ENABLE $ANS_SGD_ENABLE_VALUE \
		-env ANS_SGD_TYPE $ANS_SGD_TYPE_VALUE \
		-env ANS_SGD_PUSH_STEP $ANS_SGD_PUSH_STEP_VALUE \
		-env ANS_SGD_FETCH_STEP $ANS_SGD_FETCH_STEP_VALUE \
		-env ANS_SGD_LEARN_RATE $ANS_SGD_LEARN_RATE_VALUE \
		-env ANS_SGD_EA_ROU $ANS_SGD_EA_ROU_value \
		-env AUTHOR_SERVER_IP $AUTHOR_SERVER_IP \
		-env AUTHOR_SERVER_PORT $AUTHOR_SERVER_PORT \
		-machinefile ./hosts \
		-np $process_num \
		nohup ./build/tools/caffe.bin $PARAMETER_LIST -gpu=$GPU_LIST 2> /data/liuxu/container_qan/log/train_log.log &   #train
else
	./build/tools/caffe.bin $PARAMETER_LIST -gpu=$GPU_LIST
fi


