#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PartExtForward(const int n, const Dtype* in, Dtype* out,
                               int in_w, int in_h, int out_w, int out_h) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % out_w;
        int y = (index / out_w) % out_h;
        int map_idx = index / out_w / out_h;
        out[index] = in[(map_idx * in_h + y) * in_w + x];
    }
}

template <typename Dtype>
__global__ void PartExtForwardM(const int n, const Dtype* in, Dtype* out,
                               int in_w, int in_h, int out_w, int out_h) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % out_w;
        int y = (index / out_w) % out_h;
        int map_idx = index / out_w / out_h;
        int ox = out_w - 1 - x;
        out[(map_idx * out_h + y) * out_w + ox] = in[(map_idx * in_h + y) * in_w + x];
    }
}

template <typename Dtype>
void PartExtLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->mutable_gpu_data() + bottom[0]->offset(0, 0, crop_h_, crop_w_);
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->num();
  int ch = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->height();
  int count = num * ch * crop_w_ * crop_h_;
  if (!mirror_)
      PartExtForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, top_data, width, height, crop_w_, crop_h_);
  else
      PartExtForwardM<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, top_data, width, height, crop_w_, crop_h_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void PartExtBackward(const int n, const Dtype* in, Dtype* out,
                               int in_w, int in_h, int out_w, int out_h) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % in_w;
        int y = (index / in_w) % in_h;
        int map_idx = index / in_w / in_h;
        out[(map_idx * out_h + y) * out_w + x] = in[index];
    }
}

template <typename Dtype>
__global__ void PartExtBackwardM(const int n, const Dtype* in, Dtype* out,
                               int in_w, int in_h, int out_w, int out_h) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % in_w;
        int y = (index / in_w) % in_h;
        int map_idx = index / in_w / in_h;
        int ix = in_w - 1 - x;
        out[(map_idx * out_h + y) * out_w + x] = in[(map_idx * in_h + y) * in_w + ix];
    }
}

template <typename Dtype>
void PartExtLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff() + bottom[0]->offset(0, 0, crop_h_, crop_w_);
  const Dtype* top_diff = top[0]->gpu_diff();
  int num = bottom[0]->num();
  int ch = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->height();
  int count = num * ch * crop_w_ * crop_h_;
  if (!mirror_)
      PartExtBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, bottom_diff, crop_w_, crop_h_, width, height);
  else
      PartExtBackwardM<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, bottom_diff, crop_w_, crop_h_, width, height);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(PartExtLayer);

}  // namespace caffe
