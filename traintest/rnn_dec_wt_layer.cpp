#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/rnn_dec_wt_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RNNDecWTLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  int att_type_ = this->layer_param_.att_param().type();
  if (att_type_ > 0) {
	  if (this->phase_ == TEST) {
		  names->resize(4);
		  (*names)[0] = "h_0";
		  (*names)[1] = "c_0";
		  (*names)[2] = "y_0";
		  (*names)[3] = "softmax_0";
	  }
	  else if (this->phase_ == TRAIN) {
		  names->resize(3);
		  (*names)[0] = "h_0";
		  (*names)[1] = "c_0";
		  (*names)[2] = "softmax_0";
	  }
  }
  else {
	  if (this->phase_ == TEST) {
		  names->resize(3);
		  (*names)[0] = "h_0";
		  (*names)[1] = "c_0";
		  (*names)[2] = "y_0";
	  }
	  else if (this->phase_ == TRAIN) {
		  names->resize(2);
		  (*names)[0] = "h_0";
		  (*names)[1] = "c_0";
	  }
  }
}

template <typename Dtype>
void RNNDecWTLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
	int att_type_ = this->layer_param_.att_param().type();
	if (att_type_ > 0) {
		if (this->phase_ == TEST) {
			names->resize(4);
			(*names)[0] = "h_" + this->int_to_str(this->T_);
			(*names)[1] = "c_" + this->int_to_str(this->T_);
			(*names)[2] = "y_" + this->int_to_str(this->T_);
			(*names)[3] = "softmax_" + this->int_to_str(this->T_);
		}
		else if (this->phase_ == TRAIN) {
			names->resize(3);
			(*names)[0] = "h_" + this->int_to_str(this->T_);
			(*names)[1] = "c_" + this->int_to_str(this->T_);
			(*names)[2] = "softmax_" + this->int_to_str(this->T_);
		}
	}
	else {
		if (this->phase_ == TEST) {
			names->resize(3);
			(*names)[0] = "h_" + this->int_to_str(this->T_);
			(*names)[1] = "c_" + this->int_to_str(this->T_);
			(*names)[2] = "y_" + this->int_to_str(this->T_);
		}
		else if (this->phase_ == TRAIN) {
			names->resize(2);
			(*names)[0] = "h_" + this->int_to_str(this->T_);
			(*names)[1] = "c_" + this->int_to_str(this->T_);
		}
	}
}


template <typename Dtype>
void RNNDecWTLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
	if (this->layer_param_.att_param().has_alpha_output() && this->layer_param_.att_param().has_glimpse_output()) {
		  names->resize(3);
		  (*names)[0] = "h";
		  (*names)[1] = "alpha";
		  (*names)[2] = "glimpse";
	  }
	else if (this->layer_param_.att_param().has_hidden_output()) {
		names->resize(2);
		(*names)[0] = "h";
		(*names)[1] = "hidden";
	}
  else if (this->layer_param_.att_param().has_alpha_output()) {
	names->resize(2);
	(*names)[0] = "h";
	(*names)[1] = "alpha";
  }
  else if (this->layer_param_.att_param().has_debug_info()) {
	names->resize(6);
	(*names)[0] = "h";
	(*names)[1] = "alpha";
	(*names)[2] = "Wh_1";
	(*names)[3] = "tile_wh_1";
	(*names)[4] = "gt_swap_2";
	(*names)[5] = "g_swap_2";
  }
  else {
	names->resize(1);
	(*names)[0] = "h";
  }
  
}

template <typename Dtype>
void RNNDecWTLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";

  // attention type, 0: content, 1: location, 2: hybrid, default: 0
  int att_type_ = this->layer_param_.att_param().type();

  // begin symbol for TEST
  const int bos = this->layer_param_.att_param().bos();

  const int embed_input_dim = this->layer_param_.att_param().embed_input_dim();
  const int embed_num_output = this->layer_param_.att_param().embed_num_output();

  // location window size, default: 5
  int loc_size_ = this->layer_param_.att_param().loc_size();

  // location dimension
  int loc_dim_ = this->layer_param_.att_param().loc_dim();
  if (att_type_ > 0) {
	  CHECK_GT(loc_size_, 0) << "location window size must be positive";
	  CHECK_GT(loc_dim_, 0) << "location dimension must be positive";
  }
  
  // add coordinate info
  int concat_coordinate_ = this->layer_param_.att_param().concat_coordinate();

  const FillerParameter& weight_filler =
      this->layer_param_.recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.recurrent_param().bias_filler();

#ifndef layer_model
  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter hidden_param;
  hidden_param.set_type("InnerProduct");
  hidden_param.mutable_inner_product_param()->set_num_output(num_output);
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  hidden_param.mutable_inner_product_param()->set_axis(2);
  hidden_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
  biased_hidden_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  LayerParameter prod_param;
  prod_param.set_type("Eltwise");
  prod_param.mutable_eltwise_param()->set_operation(
	  EltwiseParameter_EltwiseOp_PROD);


  LayerParameter slice_param;
  slice_param.set_type("Slice");

  //LayerParameter reduction_param;
  //reduction_param.set_type("Reduction");

  LayerParameter tile_param;
  tile_param.set_type("Tile");

  LayerParameter split_param;
  split_param.set_type("Split");

  LayerParameter reshape_param;
  reshape_param.set_type("Reshape");

  LayerParameter permute_param;
  permute_param.set_type("Permute");
  LayerParameter conv_param;
  conv_param.set_type("Convolution");
  conv_param.mutable_convolution_param()->mutable_weight_filler()->set_type("gaussian");
  conv_param.mutable_convolution_param()->mutable_weight_filler()->set_std(0.01);
  conv_param.mutable_convolution_param()->mutable_bias_filler()->set_type("constant");
  conv_param.mutable_convolution_param()->mutable_bias_filler()->set_std(0);

  LayerParameter embed_param;
  embed_param.set_type("Embed");
  embed_param.mutable_embed_param()->set_bias_term(false);
  embed_param.mutable_embed_param()->set_input_dim(embed_input_dim);
  embed_param.mutable_embed_param()->set_num_output(embed_num_output);
  embed_param.mutable_embed_param()->mutable_weight_filler()->CopyFrom(weight_filler);

  LayerParameter argmax_param;
  argmax_param.set_type("ArgMax");
  argmax_param.add_propagate_down(false);
  
  LayerParameter concat_coordinate_param;
  concat_coordinate_param.set_type("ConcatCoordinate");
#endif

  BlobShape input_shape;
  input_shape.add_dim(1);  // c_0 and h_0 are a single timestep
  input_shape.add_dim(this->N_);
  input_shape.add_dim(num_output);

  //net_param->add_input("c_0");
  //net_param->add_input_shape()->CopyFrom(input_shape);

  //net_param->add_input("h_0");
  //net_param->add_input_shape()->CopyFrom(input_shape);
  
    LayerParameter* input_layer_param = net_param->add_layer();
	input_layer_param->set_type("Input");
	InputParameter* input_param = input_layer_param->mutable_input_param();

	input_layer_param->add_top("c_0");
	input_param->add_shape()->CopyFrom(input_shape);

	input_layer_param->add_top("h_0");
	input_param->add_shape()->CopyFrom(input_shape);

  if (this->phase_ == TEST) {
	  input_shape.Clear();
	  input_shape.add_dim(1);
	  input_shape.add_dim(this->N_);
      input_layer_param->add_top("y_0");
	  input_param->add_shape()->CopyFrom(input_shape);
  }


  // need location alpha_{t-1}
  if (att_type_ > 0) {
	  input_shape.Clear();
	  input_shape.add_dim(this->input_length_);
	  input_shape.add_dim(this->N_);
	  input_shape.add_dim(1);
	  input_layer_param->add_top("softmax_0");
	  input_param->add_shape()->CopyFrom(input_shape);
  }

  
  if (this->phase_ == TRAIN) {
	  LayerParameter *y_slice_train = net_param->add_layer();
	  // top shape [1 x B]
	  y_slice_train->CopyFrom(slice_param);
	  y_slice_train->add_bottom("label_shift");
	  y_slice_train->set_name("y_slice_train");
	  y_slice_train->mutable_slice_param()->set_axis(0);

	  for (int t = 1; t <= this->T_; ++t) {
		  string tm1s = this->int_to_str(t - 1);
		  y_slice_train->add_top("label_" + tm1s);
	  }
  }

  LayerParameter* cont_slice_param = net_param->add_layer();
  cont_slice_param->CopyFrom(slice_param);
  cont_slice_param->set_name("cont_slice");
  cont_slice_param->add_bottom("cont");
  cont_slice_param->mutable_slice_param()->set_axis(1);

  // output h
  LayerParameter output_concat_layer;
  output_concat_layer.set_name("h_concat");
  output_concat_layer.set_type("Concat");
  output_concat_layer.add_top("h");
  output_concat_layer.mutable_concat_param()->set_axis(0);


  /**
  * bottom[0]: alpha			(input_length x N x T_)
  * bottom[1]: "wh_"			(1 x N x D)
  * bottom[2]: "tile_wh_"		(input_length x N x D)
  * bottom[3]: "gt_"			(D x N x input_length)
  * bottom[4]: "g_"			(D x N)
  */

  LayerParameter output_alpha_layer;
  if (this->layer_param_.att_param().has_alpha_output() || this->layer_param_.att_param().has_debug_info()) {
	  // output alpha: input_length x N x T_
	  output_alpha_layer.set_name("alpha_concat");
	  output_alpha_layer.set_type("Concat");
	  output_alpha_layer.add_top("alpha");
	  output_alpha_layer.mutable_concat_param()->set_axis(2);
  }
  LayerParameter output_glimpse_layer;
  if (this->layer_param_.att_param().has_glimpse_output()) {
	  // output alpha: input_length x N x T_
	  output_glimpse_layer.set_name("glimpse_concat");
	  output_glimpse_layer.set_type("Concat");
	  output_glimpse_layer.add_top("glimpse");
	  output_glimpse_layer.mutable_concat_param()->set_axis(0);
  }

  LayerParameter output_hidden_layer;
  if (this->layer_param_.att_param().has_hidden_output()) {
	  // output hidden: input_length x N x T_
	  output_hidden_layer.set_name("hidden_concat");
	  output_hidden_layer.set_type("Concat");
	  output_hidden_layer.add_top("hidden");
	  output_hidden_layer.mutable_concat_param()->set_axis(0);
  }
  
  /****************************** Begin: Add Attention operation **********************************/
  // concat coordinate info to x
  if (concat_coordinate_ == 1)
  {
	  LayerParameter* concat_coord = net_param->add_layer();
	  concat_coord->set_name("concat_coord");
	  concat_coord->CopyFrom(concat_coordinate_param);
	  concat_coord->mutable_concat_coordinate_param()->set_concat_coordinate_method(0);
	  concat_coord->mutable_concat_coordinate_param()->set_length_axis(0);
	  concat_coord->mutable_concat_coordinate_param()->set_channel_axis(2);
	  //concat_coord->mutable_concat_coordinate_param()->set_rows_num(1);
	  //concat_coord->mutable_concat_coordinate_param()->set_length_axis2(0);
	  concat_coord->add_bottom("x");
	  concat_coord->add_top("x_coord");
  }


  // Add layer to transform all timesteps of x to the hidden state dimension.
  //     W_xc_x = W_xc * x + b_c
  {
    LayerParameter* x_transform_param = net_param->add_layer();
    x_transform_param->CopyFrom(biased_hidden_param);
	x_transform_param->mutable_inner_product_param()->set_num_output(num_output);
    x_transform_param->set_name("x_transform");
    x_transform_param->add_param()->set_name("W_xc");
    x_transform_param->add_param()->set_name("b_c");
    x_transform_param->add_bottom("x");
    x_transform_param->add_top("W_xc_x");
    x_transform_param->add_propagate_down(true);
  }

  for (int t = 1; t <= this->T_; ++t) {
	  string tm1s = this->int_to_str(t - 1);
	  string ts = this->int_to_str(t);
#if 1
	
	  // compute location component
	  if (att_type_ > 0) {
		  // Reshape [Shape: input_length_ x N x 1 x 1]
		  LayerParameter* reshape_loc = net_param->add_layer();
		  reshape_loc->CopyFrom(reshape_param);
		  reshape_loc->set_name("reshape_loc_" + tm1s);
		  reshape_loc->mutable_reshape_param()->mutable_shape()->add_dim(this->input_length_);
		  reshape_loc->mutable_reshape_param()->mutable_shape()->add_dim(this->N_);
		  reshape_loc->mutable_reshape_param()->mutable_shape()->add_dim(1);
		  reshape_loc->mutable_reshape_param()->mutable_shape()->add_dim(1);
		  reshape_loc->add_bottom("softmax_" + tm1s);
		  reshape_loc->add_top("a_pre_" + tm1s);

		  // permute: N_ x 1 x 1 x inputlength_
		  LayerParameter* permute_loc = net_param->add_layer();
		  permute_loc->set_name("permute_loc_" + tm1s);
		  permute_loc->CopyFrom(permute_param);
		  permute_loc->mutable_permute_param()->add_order(1);
		  permute_loc->mutable_permute_param()->add_order(2);
		  permute_loc->mutable_permute_param()->add_order(3);
		  permute_loc->mutable_permute_param()->add_order(0);
		  permute_loc->add_bottom("a_pre_" + tm1s);
		  permute_loc->add_top("a_" + tm1s);

		  // Convolution [Shape: N_ x loc_dim x 1 x inputlength_]
		  // Here we conv \alpha_{t-1} for computing the location info
		  // Location window size:		loc_size_ 
		  // Dimension size:			loc_dim_
		  // Formula:					f_t = F * \alpha_{t-1}
		  LayerParameter* conv_alpha = net_param->add_layer();
		  conv_alpha->set_name("conv_alpha");
		  conv_alpha->CopyFrom(conv_param);
		  conv_alpha->mutable_convolution_param()->set_kernel_h(1);
		  conv_alpha->mutable_convolution_param()->set_kernel_w(loc_size_);
		  conv_alpha->mutable_convolution_param()->set_num_output(loc_dim_);
		  conv_alpha->mutable_convolution_param()->set_pad_w(loc_size_ / 2);
		  conv_alpha->add_param()->set_name("conv_alpha_");
		  conv_alpha->add_param()->set_name("conv_alpha_bias_");
		  conv_alpha->add_bottom("a_" + tm1s);
		  conv_alpha->add_top("f_" + tm1s);

		  // Reshape [Shape: N_ x loc_dim x inputlength_]
		  LayerParameter* reshape_loc1 = net_param->add_layer();
		  reshape_loc1->CopyFrom(reshape_param);
		  reshape_loc1->set_name("reshape_loc1_" + tm1s);
		  reshape_loc1->mutable_reshape_param()->mutable_shape()->add_dim(this->N_);
		  reshape_loc1->mutable_reshape_param()->mutable_shape()->add_dim(loc_dim_);
		  reshape_loc1->mutable_reshape_param()->mutable_shape()->add_dim(this->input_length_);
		  reshape_loc1->add_bottom("f_" + tm1s);
		  reshape_loc1->add_top("f_after_" + tm1s);

		  // Permute [Shape: inputlength_ x N_ x loc_dim_]
		  LayerParameter* permute_loc1 = net_param->add_layer();
		  permute_loc1->set_name("permute_loc1_" + tm1s);
		  permute_loc1->CopyFrom(permute_param);
		  permute_loc1->mutable_permute_param()->add_order(2);
		  permute_loc1->mutable_permute_param()->add_order(0);
		  permute_loc1->mutable_permute_param()->add_order(1);
		  permute_loc1->add_bottom("f_after_" + tm1s);
		  permute_loc1->add_top("f_final_" + tm1s);

		  // InnerProduct for Location [Shape: inputlength_ x N x D]
		  LayerParameter* walpha_param = net_param->add_layer();
		  walpha_param->CopyFrom(hidden_param);
		  walpha_param->set_name("walpha_transform");
		  walpha_param->mutable_inner_product_param()->set_num_output(num_output);
		  walpha_param->add_param()->set_name("Walpha_");
		  walpha_param->add_bottom("f_final_" + tm1s);
		  walpha_param->add_top("walpha_" + tm1s); // 1 x N x D
	  }

	  // compute content component
	  if (att_type_ != 1) {
		  // InnerProduct [Shape: 1 x N x D]
		  LayerParameter* wh_param = net_param->add_layer();
		  wh_param->CopyFrom(hidden_param);
		  wh_param->set_name("wh_transform");
		  wh_param->mutable_inner_product_param()->set_num_output(num_output);
		  wh_param->add_param()->set_name("Wh_");
		  //wh_param->add_param()->set_name("bh_");
		  wh_param->add_bottom("h_" + tm1s);
		  wh_param->add_top("Wh_" + tm1s); 

		  // Tile [Shape: input_length x N x D]
		  LayerParameter* tile_0 = net_param->add_layer();
		  tile_0->set_name("tile_wh_" + ts);
		  tile_0->CopyFrom(tile_param);
		  tile_0->mutable_tile_param()->set_axis(0);
		  tile_0->mutable_tile_param()->set_tiles(this->input_length_);
		  tile_0->add_bottom("Wh_" + tm1s);
		  tile_0->add_top("tile_wh_" + tm1s);
	  }

	// Sum [Shape: input_length x N x D]

	LayerParameter* e_sum_layer = net_param->add_layer();
	e_sum_layer->CopyFrom(sum_param);
	e_sum_layer->set_name("hat_e_" + ts);
	e_sum_layer->add_bottom("W_xc_x"); 
	if (att_type_ == 0) {
		// \hat_{e} = Wh_{t-1} + Vx_j + b   ----------------> ref[2] eq(7)
		e_sum_layer->add_bottom("tile_wh_" + tm1s);
	}
	else if (att_type_ == 1) {
		e_sum_layer->add_bottom("walpha_" + tm1s);
	}
	else {
		// \hat_{e} = Wh_{t-1} + Vx_j + Uf_{t,j} + b   ----------------> ref[2] eq(9)
		e_sum_layer->add_bottom("tile_wh_" + tm1s);
		e_sum_layer->add_bottom("walpha_" + tm1s);
	}
	e_sum_layer->add_top("hat_e_" + ts);

	// Tanh [Shape: input_length x N x D]
	LayerParameter *e_tanh_layer = net_param->add_layer();
	e_tanh_layer->set_name("hat_e_tanh_" + ts);
	e_tanh_layer->set_type("TanH");
	e_tanh_layer->add_bottom("hat_e_" + ts);
	e_tanh_layer->add_top("hat_e_tanh_" + ts);

	// InnerProduct [Shape: input_length x N x 1]
	LayerParameter* v_param = net_param->add_layer();
	v_param->CopyFrom(hidden_param);
	v_param->set_name("e_" + ts);
	v_param->add_param()->set_name("v_");
	v_param->add_bottom("hat_e_tanh_" + ts);
	v_param->mutable_inner_product_param()->set_num_output(1);
	v_param->add_top("e_" + ts);

	// Softmax [Shape: input_length x N x 1]
	// Formula: \alpha_{t,j} = Softmax(e_{t,j})  ----------------> ref[2] eq(6)
	LayerParameter* softmax = net_param->add_layer();
	softmax->set_type("Softmax");
	softmax->set_name("softmax_" + ts);
	softmax->add_bottom("e_" + ts);
	softmax->mutable_softmax_param()->set_axis(0);
	softmax->add_top("softmax_" + ts);

	if (this->layer_param_.att_param().has_alpha_output() || this->layer_param_.att_param().has_debug_info()) {
		output_alpha_layer.add_bottom("softmax_" + ts);
	}

	// Tile [Shape: input_length x N x D]
	LayerParameter* tile_2 = net_param->add_layer();
	tile_2->set_name("tile_" + ts);
	tile_2->CopyFrom(tile_param);
	tile_2->mutable_tile_param()->set_axis(2);
	tile_2->mutable_tile_param()->set_tiles(num_output);
	tile_2->add_bottom("softmax_" + ts);
	tile_2->add_top("t_softmax_" + ts);

	// Element-wise Product x and g [Shape: input_length x N x D]
	// Formula: g_t = \sum_{j=1}^T \alpha_{t,j}h_j
	LayerParameter* g_prod_layer = net_param->add_layer();
	g_prod_layer->CopyFrom(prod_param);
	g_prod_layer->set_name("gt_" + ts);
	g_prod_layer->add_bottom("t_softmax_" + ts);
	g_prod_layer->add_bottom("x");
	g_prod_layer->add_top("gt_" + ts);
	// Reduction [Shape: 1 x N x D]
	LayerParameter* g_reduction_2 = net_param->add_layer();
	g_reduction_2->set_name("g_" + ts);
	g_reduction_2->set_type("ComputeAxis");
	g_reduction_2->mutable_compute_axis_param()->set_axis(0);
	g_reduction_2->add_bottom("gt_" + ts);
	g_reduction_2->add_top("g_" + ts);
	if (this->layer_param_.att_param().has_glimpse_output()) {
		output_glimpse_layer.add_bottom("g_" + ts);
	}
	// Add layer to transform all timesteps of gto the hidden state dimension.
	//     W_gc_g = W_gc * g + b_c
	{
		LayerParameter* g_transform_param = net_param->add_layer();
		g_transform_param->CopyFrom(biased_hidden_param);
		g_transform_param->mutable_inner_product_param()->set_num_output(num_output * 4);
		g_transform_param->set_name("g_transform" + ts);
		g_transform_param->add_param()->set_name("Wg_gc");
		g_transform_param->add_param()->set_name("bg_c");
		g_transform_param->add_bottom("g_" + ts);
		//g_transform_param->add_bottom("test_g");
		g_transform_param->add_top("W_gc_g_" + ts);
		g_transform_param->add_propagate_down(true);
	}
#endif

  /****************************** End: Add Attention operation   **********************************/
#if 1 // LSTM modeule
	cont_slice_param->add_top("cont_" + ts);

	if (this->phase_ == TEST) {
		LayerParameter* test_label_copy_param = net_param->add_layer();
		test_label_copy_param->CopyFrom(split_param);
		test_label_copy_param->set_name("test_y_" + tm1s);
		test_label_copy_param->add_bottom("y_" + tm1s);
		test_label_copy_param->add_top("label_" + tm1s);
	}

	LayerParameter* y_embed_param = net_param->add_layer();
	y_embed_param->CopyFrom(embed_param);
	y_embed_param->add_param()->set_name("W_y_embed");
	y_embed_param->set_name("embed_y_" + ts);
	y_embed_param->add_bottom("label_" + tm1s);
	y_embed_param->add_top("embed_y_" + ts);
	
	// Add layer to transform all timesteps of embed y to the word state dimension.
	//     W_embed_y = W_embed * y + b_c
	{
		LayerParameter* y_transform_param = net_param->add_layer();
		y_transform_param->CopyFrom(hidden_param);
		y_transform_param->mutable_inner_product_param()->set_num_output(num_output * 4);
		y_transform_param->set_name("y_transform_" + ts);
		y_transform_param->add_param()->set_name("W_embed");
		y_transform_param->add_bottom("embed_y_" + ts);
		y_transform_param->add_top("W_embed_y_" + ts);
		y_transform_param->add_propagate_down(true);
	}

#if 1
	// Add layers to flush the hidden state when beginning a new
	// sequence, as indicated by cont_t.
	//     h_conted_{t-1} := cont_t * h_{t-1}
	//
	// Normally, cont_t is binary (i.e., 0 or 1), so:
	//     h_conted_{t-1} := h_{t-1} if cont_t == 1
	//                       0   otherwise
	{
		LayerParameter* cont_h_param = net_param->add_layer();
		cont_h_param->CopyFrom(sum_param);
		cont_h_param->mutable_eltwise_param()->set_coeff_blob(true);
		cont_h_param->set_name("h_conted_" + tm1s);
		cont_h_param->add_bottom("h_" + tm1s);
		cont_h_param->add_bottom("cont_" + ts);
		cont_h_param->add_top("h_conted_" + tm1s);
	}

	// Add layer to compute
	//     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
	{
		LayerParameter* w_param = net_param->add_layer();
		w_param->CopyFrom(hidden_param);
		w_param->mutable_inner_product_param()->set_num_output(num_output * 4);
		w_param->set_name("transform_" + ts);
		w_param->add_param()->set_name("W_hc");
		w_param->add_bottom("h_conted_" + tm1s);
		w_param->add_top("W_hc_h_" + tm1s);
		w_param->mutable_inner_product_param()->set_axis(2);
	}

    // Add the outputs of the linear transformations to compute the gate input.
    //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
    {
      LayerParameter* input_sum_layer = net_param->add_layer();
      input_sum_layer->CopyFrom(sum_param);
      input_sum_layer->set_name("gate_input_" + ts);
      input_sum_layer->add_bottom("W_hc_h_" + tm1s);
	  input_sum_layer->add_bottom("W_gc_g_" + ts);
	  input_sum_layer->add_bottom("W_embed_y_" + ts);
      input_sum_layer->add_top("gate_input_" + ts);
    }

    // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
    // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
    // Outputs: c_t, h_t
    //     [ i_t' ]
    //     [ f_t' ] := gate_input_t
    //     [ o_t' ]
    //     [ g_t' ]
    //         i_t := \sigmoid[i_t']
    //         f_t := \sigmoid[f_t']
    //         o_t := \sigmoid[o_t']
    //         g_t := \tanh[g_t']
    //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    //         h_t := o_t .* \tanh[c_t]
    {
      LayerParameter* lstm_unit_param = net_param->add_layer();
      lstm_unit_param->set_type("RNNDecUnit");
      lstm_unit_param->add_bottom("c_" + tm1s);
      lstm_unit_param->add_bottom("gate_input_" + ts);
      lstm_unit_param->add_bottom("cont_" + ts);
      lstm_unit_param->add_top("c_" + ts);
      lstm_unit_param->add_top("h_" + ts);
      lstm_unit_param->set_name("unit_" + ts);

	  if (this->layer_param_.att_param().has_hidden_output()) {
		  output_hidden_layer.add_bottom("h_" + ts);
	  }
    }
#endif
	// 1 x B x num_output
	{
		LayerParameter* w_param = net_param->add_layer();
		w_param->CopyFrom(hidden_param);
		w_param->mutable_inner_product_param()->set_num_output(embed_input_dim);
		w_param->set_name("transform_h_" + ts);
		w_param->add_param()->set_name("W_h_after");
		w_param->add_bottom("h_" + ts);
		w_param->add_top("h_output_" + ts);
		w_param->mutable_inner_product_param()->set_axis(2);
	}
	if (this->phase_ == TEST) {
		// 1 x B x num_output
		LayerParameter* softmax_h = net_param->add_layer();
		softmax_h->set_type("Softmax");
		softmax_h->set_name("softmax_h_" + ts);
		softmax_h->add_bottom("h_output_" + ts);
		softmax_h->mutable_softmax_param()->set_axis(2);
		softmax_h->add_top("h_softmax_" + ts);
		softmax_h->add_propagate_down(false);

		// 1 x B
		LayerParameter *argmax_h = net_param->add_layer();
		argmax_h->CopyFrom(argmax_param);
		argmax_h->set_name("y_" + ts);
		argmax_h->mutable_argmax_param()->set_axis(2);
		argmax_h->add_bottom("h_softmax_" + ts);
		argmax_h->add_top("y_" + ts);
	}
    //output_concat_layer.add_bottom("h_" + ts);
	output_concat_layer.add_bottom("h_output_" + ts);
#endif
  }  // for (int t = 1; t <= this->T_; ++t)

  //{
  //  LayerParameter* c_T_copy_param = net_param->add_layer();
  //  c_T_copy_param->CopyFrom(split_param);
  //  c_T_copy_param->add_bottom("c_" + format_int(this->T_));
  //  c_T_copy_param->add_top("c_T");
  //}
  
  net_param->add_layer()->CopyFrom(output_concat_layer);
  if (this->layer_param_.att_param().has_alpha_output() || this->layer_param_.att_param().has_debug_info()) {
	  net_param->add_layer()->CopyFrom(output_alpha_layer);
  }
  if (this->layer_param_.att_param().has_glimpse_output()) {
	  net_param->add_layer()->CopyFrom(output_glimpse_layer);
  }
  if (this->layer_param_.att_param().has_hidden_output()) {
	  net_param->add_layer()->CopyFrom(output_hidden_layer);
  }
}

INSTANTIATE_CLASS(RNNDecWTLayer);
REGISTER_LAYER_CLASS(RNNDecWT);

}  // namespace caffe
