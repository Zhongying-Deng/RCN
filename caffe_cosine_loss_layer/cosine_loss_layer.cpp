#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cosine_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CosineLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.cosine_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  
}

template <typename Dtype>
void CosineLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), bottom[0]->channels());
  CHECK_EQ(bottom[1]->num(), bottom[0]->num());
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  M_ = bottom[0]->num();
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void CosineLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data_0 = bottom[0]->cpu_data();
  const Dtype* bottom_data_1 = bottom[1]->cpu_data();
  Dtype loss = caffe_cpu_dot(M_ * K_, bottom_data_0, bottom_data_1);
  top[0]->mutable_cpu_data()[0] = Dtype(1.0) - loss / M_;
  if(loss / M_ >= 1){
  LOG(INFO)<< "batch size:" << M_ << ",feature dim:" << K_ << ",loss:" << loss;}
}

template <typename Dtype>
void CosineLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data_0 = bottom[0]->cpu_data();
  const Dtype* bottom_data_1 = bottom[1]->cpu_data();
  Dtype* bottom_diff_0 = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_diff_1 = bottom[1]->mutable_cpu_diff();
  Dtype scale = - top[0]->cpu_diff()[0] / Dtype(M_);
  //LOG(INFO)<< "scale:" << scale << ",scale_:" << - top[0]->cpu_diff()[0] / M_;
  
  // Gradient with respect to bottom data 
  if (propagate_down[0]) {
    caffe_copy(M_ * K_, bottom_data_1, bottom_diff_0);
    caffe_scal(M_ * K_, scale, bottom_diff_0);
  }
  if (propagate_down[1]) {
    caffe_copy(M_ * K_, bottom_data_0, bottom_diff_1);
    caffe_scal(M_ * K_, scale, bottom_diff_1);
  }
}

#ifdef CPU_ONLY
STUB_GPU(CosineLossLayer);
#endif

INSTANTIATE_CLASS(CosineLossLayer);
REGISTER_LAYER_CLASS(CosineLoss);

}  // namespace caffe
