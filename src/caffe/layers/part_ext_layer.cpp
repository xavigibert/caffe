#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PartExtLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PartExtLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  const PartExtParameter& part_ext_param = this->layer_param_.part_ext_param();
  off_x_ = int(part_ext_param.off_x() * width);
  off_y_ = int(part_ext_param.off_y() * height);
  crop_w_ = int(part_ext_param.crop_w() * width);
  crop_h_ = int(part_ext_param.crop_h() * height);
  CHECK_GE(off_x_, 0);
  CHECK_GE(off_y_, 0);
  CHECK_LE(off_x_ + crop_w_, width) << "Crop region too wide";
  CHECK_LE(off_y_ + crop_h_, height) << "Crop region too tall";

  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), crop_h_, crop_w_);
}

template <typename Dtype>
void PartExtLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int ch = bottom[0]->channels();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < ch; ++c) {
      for (int y = 0; y < crop_h_; ++y) {
        const Dtype* src = bottom_data + bottom[0]->offset(n, c, y + off_y_, off_x_);
        Dtype* dst = top_data + top[0]->offset(n, c, y, 0);
        if (!mirror_)
          caffe_copy(crop_w_, src, dst);
        else {
          for (int x = 0; x < crop_w_; ++x) {
              dst[x] = src[crop_w_ - x - 1];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void PartExtLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  int num = bottom[0]->num();
  int ch = bottom[0]->channels();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < ch; ++c) {
      for (int y = 0; y < crop_h_; ++y) {
        const Dtype* src = top_diff + top[0]->offset(n, c, y, 0);
        Dtype* dst = bottom_diff + bottom[0]->offset(n, c, y + off_y_, off_x_);
        if (!mirror_)
          caffe_copy(crop_w_, src, dst);
        else {
          for (int x = 0; x < crop_w_; ++x) {
              dst[x] = src[crop_w_ - x - 1];
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PartExtLayer);
#endif

INSTANTIATE_CLASS(PartExtLayer);
REGISTER_LAYER_CLASS(PartExt);

}  // namespace caffe
