#include <algorithm>
#include <vector>

#include "caffe/layers/seq_mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	void SeqMaskLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	}

	template <typename Dtype>
	void SeqMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
        vector<int> cur_shape;
        cur_shape.push_back(bottom[0]->shape(0));
        cur_shape.push_back(bottom[0]->shape(1));
        top[0]->Reshape(cur_shape);
	}

	template <typename Dtype>
	void SeqMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* label_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		int batch_size = bottom[0]->shape(1);
		int input_length = bottom[0]->shape(0);
		const int eos = this->layer_param_.att_param().eos();
		//std::string str_map = "_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
		// begin: 36, end: 37, other 38
		for (int i = 0; i < batch_size; i++) {
			for (int j = 0; j < input_length; j++) {
				if (this->phase_ == TRAIN) {
					if (label_data[j * batch_size + i] == eos) {
						top_data[j * batch_size + i] = 0;
						//std::cout << str_map[label_data[j * batch_size + i]];
					}
					else {
						top_data[j * batch_size + i] = 1;
						//std::cout << str_map[label_data[j * batch_size + i]];
						//std::cout << top_data[j * batch_size + i] << std::endl;
					}
				}
				else if (this->phase_ == TEST) {
					top_data[j * batch_size + i] = 1;
				}
				top_data[j * batch_size + i] = 1;
			}
			//std::cout << std::endl;
		}
	}

	template <typename Dtype>
	void SeqMaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		// No need to backward
	}


#ifdef CPU_ONLY
	STUB_GPU(SeqMaskLayer);
#endif

	INSTANTIATE_CLASS(SeqMaskLayer);
	REGISTER_LAYER_CLASS(SeqMask);
}  // namespace caffe
