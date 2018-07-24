#include <algorithm>
#include <vector>

#include "caffe/layers/shift_seq_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	void ShiftSeqLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	}

	template <typename Dtype>
	void ShiftSeqLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
        vector<int> cur_shape;
        cur_shape.push_back(bottom[0]->shape(0));
        cur_shape.push_back(bottom[0]->shape(1));
        top[0]->Reshape(cur_shape);
	}

	template <typename Dtype>
	void ShiftSeqLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* label_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

        const int bos = this->layer_param_.att_param().bos();

		int batch_size = bottom[0]->shape(1);
		int input_length = bottom[0]->shape(0);
		//std::string str_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ$_#";
		// begin: 36, other: 37, end: 38
		for (int i = 0; i < batch_size; i++) {
			for (int j = 0; j < input_length; j++) {
				// 第一个字符初始化为36，即起始符BOS
				if (j == 0) {
					top_data[j * batch_size + i] = bos;
				}
				else {
					top_data[j * batch_size + i] = label_data[(j - 1) * batch_size + i];
				}

			} // for: input_length
		} // for: batch_size
#if 0
		const Dtype* dtop_data = top[0]->mutable_cpu_data();
		for (int i = 0; i < batch_size; i++) {
			for (int j = 0; j < input_length; j++) {
				for (int c = 0; c < class_num_; c++) {
					std::cout << dtop_data[j * batch_size + i + c] << ",";
				}
				std::cout << std::endl;
			} // for: input_length
			break;
		} // for: batch_size
#endif
	}

	template <typename Dtype>
	void ShiftSeqLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		// No need to backward
	}


#ifdef CPU_ONLY
	STUB_GPU(ShiftSeqLayer);
#endif

	INSTANTIATE_CLASS(ShiftSeqLayer);
	REGISTER_LAYER_CLASS(ShiftSeq);
}  // namespace caffe
