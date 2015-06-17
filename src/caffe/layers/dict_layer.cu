#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define EPSILON (1e-6)

namespace caffe {

template <typename Dtype>
__global__ void kernel_norm(int m, int k, const Dtype* D, Dtype* diagDtD) {
  CUDA_KERNEL_LOOP(index, k) {
    Dtype res = (Dtype)0.;
    for (int j =0; j < m; ++j)
      res += D[index+j*k] * D[index+j*k];
    diagDtD[index] = res;
  }
}

template <typename Dtype>
void DictionaryLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
//  // Perform normalization and decomposition (if necessary)
//  forward_preprocess_cpu();
//  // Perform sparse coding (and optionally dictionary learning) on each input vector
//  const Dtype* D = this->blobs_[0]->gpu_data();
//  const Dtype* Vt = this->blobs_[bias_idx_ + 2]->gpu_data();
//  const Dtype* Vt_s2 = this->blobs_[bias_idx_ + 3]->gpu_data();
//  for (int i = 0; i < top.size()/2; ++i) {
//    const Dtype* bottom_data = bottom[i]->gpu_data();
//    Dtype* top_data = top[2*i]->mutable_gpu_data();
//    Dtype* loss = top[2*i+1]->mutable_gpu_data();
//    caffe_gpu_set(1, (Dtype)0., loss);
//    for (int n = 0; n < this->num_; ++n) {
//      // Perform forward sparse coding
//      this->forward_gpu_sparse_coding(bottom_data + bottom[i]->offset(n),
//          D, Vt, Vt_s2, top_data + top[2*i]->offset(n), loss);
//      if (this->bias_term_) {
//        const Dtype* bias = this->blobs_[bias_idx_]->gpu_data();
//        this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
//      }
//    }
//    // Scale objective value in second output
//    caffe_gpu_scal(1, (Dtype)1./(Dtype)(num_*conv_out_spatial_dim_), loss);
//  }
}

// A is mxk, B is kxm
template <typename Dtype>
__global__ void transpose_kernel(int n, int m, int k, const Dtype* A, Dtype* B) {
  CUDA_KERNEL_LOOP(index, n) {
    int idx_row = index / m;
    int idx_col = index % m;
    B[index] = A[idx_row + idx_col*k];
  }
}

// Perform B = A^T
template<typename Dtype>
void DictionaryLayer<Dtype>::transpose_gpu(int m, int k, const Dtype* A, Dtype* B) {
  int N = m*k;
  transpose_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, m, k, A, B);
}

template <typename Dtype>
__global__ void kernel_compute_diag_weights(int n, Dtype lambda,
      const Dtype* vec_alpha, Dtype* diag) {
  CUDA_KERNEL_LOOP(index, n) {
    diag[index] = 2*lambda/(fabs(vec_alpha[index])+EPSILON);
  }
}

template <typename Dtype>
__global__ void kernel_hard_threshold(int n, Dtype lambda, Dtype* vec_alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    if (fabs(vec_alpha[index]) < lambda)
      vec_alpha[index] = (Dtype)0.;
  }
}

// Perform sparse coding on the GPU, estimate the loss and add it to the
// previous loss value
template <typename Dtype>
void DictionaryLayer<Dtype>::forward_gpu_sparse_coding(const Dtype* input,
      const Dtype* D, const Dtype* Vt, const Dtype *Vt_s2, Dtype* output,
      Dtype* loss, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  // Perform sparse coding for each input vector
  int m = kernel_dim_;
  int k = num_output_;
  Dtype* vec_d = vec_d_buffer_.mutable_gpu_data();      // D^T * x
  Dtype* vec_r = vec_r_buffer_.mutable_gpu_data();      // Residual vector
  Dtype* vec_p = vec_p_buffer_.mutable_gpu_data();      // Descent direction
  Dtype* vec_w = vec_w_buffer_.mutable_gpu_data();      // Vector w
  Dtype* sparse_codes = sparse_codes_buffer_.mutable_gpu_data();
  Dtype* diag = tmp_buffer_.mutable_gpu_data() + std::max(k,m);
  Dtype* tmp = tmp_buffer_.mutable_gpu_data() + 2*std::max(k,m);
  for (int i = 0; i < conv_out_spatial_dim_; ++i)
  {
    const Dtype* x = col_buff + i*m;  // Input sample
    Dtype* vec_alpha = sparse_codes + i*k;   // Sparse code vector
    // Initialize sparse code
    caffe_gpu_set<Dtype>(k, (Dtype)1., vec_alpha);
    // Perform num_iter_irls iterations of iteratively reweighted
    // least squares using the previous result as starting value
    for (int iter_irls = 0; iter_irls < num_iter_irls_; ++iter_irls)
    {
      // Build matrix w = diag(2*lambda/fabs(alpha[])
      kernel_compute_diag_weights<Dtype><<<CAFFE_GET_BLOCKS(k),
          CAFFE_CUDA_NUM_THREADS>>>(k, (Dtype)lambda_, vec_alpha, diag);
      // Build vector d = D^T * x
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, k, 1, m,
           (Dtype)1., D, x,
           (Dtype)0., vec_d);
      // Perform conjugate gradient descent to approximately solve
      // C * alpha = d     for alpha
      // Note: We do not compute matrix C explicitly, since
      //   C * alpha = diag(2*lambda/fabs(alpha[]) .* alpha + V * Vt_s2 * alpha
      //   C * alpha = tmp + V * Vt_s2 * alpha
      // is more efficient to compute
      conjugate_gradient_gpu(k, rank_, diag, Vt, Vt_s2, vec_d, vec_alpha,
          num_iter_cg_, vec_p, vec_r, vec_w, tmp);
    }
    // Apply hard threshold to sparse codes
    kernel_hard_threshold<<<CAFFE_GET_BLOCKS(k), CAFFE_CUDA_NUM_THREADS>>>(
        k, (Dtype)lambda_, vec_alpha);
    //add_objective_gpu(m, k, D, x, vec_alpha, loss);
  }
  // Sparse codes are in pixel-first order, we need to transpose them so they
  // are in channel-first order
  transpose_gpu(conv_out_spatial_dim_, num_output_, sparse_codes, output);
}

template <typename Dtype>
void DictionaryLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  int k = num_output_;
  caffe_gpu_add(k, bias, output, output);
}

template <typename Dtype>
void DictionaryLayer<Dtype>::add_objective_gpu(int m, int k,
        const Dtype* D, const Dtype* x, const Dtype* alpha,
        Dtype* loss) {
  // Compute objective function
  // Cost(alpha) = 0.5*||x_t-D*alpha||_2^2 + lambda*||alpha||_1
  Dtype* tmp = tmp_buffer_.mutable_gpu_data();
  caffe_gpu_memcpy(m, x, tmp);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, 1, k,
                        (Dtype)(-1.), D, alpha,
                        (Dtype)1., tmp);
  Dtype cost = (Dtype)0.;
  //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, 1, m, (Dtype)1., tmp,
  //    tmp, (Dtype)0., cost);
  caffe_gpu_dot<Dtype>(m, tmp, tmp, &cost);
  //caffe_gpu_scale<Dtype>(1, 0.5, cost, cost);
  Dtype asum = (Dtype)0.;
  // Sum of absolute values of elements in alpha
  caffe_gpu_asum<Dtype>(k, alpha, &asum);
  //caffe_gpu_scale<Dtype>(1, Dtype(lambda_), asum, asum);
  // Add cost and asum to loss
  //caffe_gpu_add<Dtype>(1, asum, loss, loss);
  //caffe_gpu_add<Dtype>(1, cost, loss, loss);
  cost = 0.5 * cost + lambda_ * asum;
  //LOG(INFO) << "COST = " << cost;
  caffe_gpu_add_scalar(1, cost, loss);
}

// Compute C * x = w .* x + V * Vt2 * x
//         (kx1)    (kx1)  (k*r)(r*k)(k*1)
template <typename Dtype>
void DictionaryLayer<Dtype>::compute_Cx_gpu(int k, int r, const Dtype* w,
      const Dtype* Vt, const Dtype* Vt2, const Dtype* x,
      Dtype* tmp, Dtype* Cx) {
  caffe_gpu_mul(k, w, x, Cx);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, r, k, (Dtype)1., Vt2, x, (Dtype)0.,
      tmp);
  caffe_gpu_gemv<Dtype>(CblasTrans, r, k, (Dtype)1., Vt, tmp, (Dtype)1., Cx);
}

template <typename Dtype>
void DictionaryLayer<Dtype>::conjugate_gradient_gpu(int k, int r,
      const Dtype* weights, const Dtype* Vt, const Dtype* Vt2, const Dtype* d,
      Dtype* x, int num_iter, Dtype* temp_p, Dtype* temp_r, Dtype* temp_w,
      Dtype* tmp) {
  // Initialize the residual
  compute_Cx_gpu(k, r, weights, Vt, Vt2, x, tmp, temp_r);
  caffe_gpu_sub(k, d, temp_r, temp_r);
  // Compute norm of the residual
  Dtype prev_norm_r = (Dtype)0.;
  caffe_gpu_dot<Dtype>(k, temp_r, temp_r, &prev_norm_r);
  if (fabs(prev_norm_r) < EPSILON) {
    return;   // Accept initial solution
  }
  // Initialize the descent direction
  caffe_gpu_memcpy(k*sizeof(Dtype), temp_r, temp_p);
  // Perform num_iter_cg iterations of conjugate gradient descent
  for (int iter_cg = 0; iter_cg < num_iter; ++iter_cg)
  {
    // w = C * p
    compute_Cx_gpu(k, r, weights, Vt, Vt2, temp_p, tmp, temp_w);
    Dtype dot_p_w = (Dtype)0.;
    caffe_gpu_dot<Dtype>(k, temp_p, temp_w, &dot_p_w);
    Dtype alpha = prev_norm_r / dot_p_w;
    CHECK(!isnan(alpha));
    // x = x + alpha*p
    caffe_gpu_axpy<Dtype>(k, alpha, temp_p, x);
    // r = r - alpha*w
    caffe_gpu_axpy<Dtype>(k, -alpha, temp_w, temp_r);
    // Compute norm of new residual
    Dtype norm_r = (Dtype)0.;
    caffe_gpu_dot<Dtype>(k, temp_r, temp_r, &norm_r);
    // Compute beta
    Dtype beta = norm_r / prev_norm_r;
    CHECK(!isnan(beta));
    // p = r + beta*p
    caffe_gpu_axpby<Dtype>(k, (Dtype)1., temp_r, beta, temp_p);
    prev_norm_r = norm_r;
    if (fabs(prev_norm_r) < EPSILON) {
      return;
    }
  }
}

template <typename Dtype>
__global__ void kernel_mod_gradient(int k, const Dtype* alpha,
    const Dtype* alpha_diff, Dtype* mod_alpha_diff) {
  CUDA_KERNEL_LOOP(index, k) {
    mod_alpha_diff[index] = alpha[index] == (Dtype)0.
        ? (Dtype)0. : alpha_diff[index];
  }
}

template <typename Dtype>
void DictionaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//  Backward_cpu(top, propagate_down, bottom);
  const Dtype* D = this->blobs_[0]->gpu_data();
  Dtype* D_diff = this->blobs_[0]->mutable_gpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), D_diff);
  }
  CHECK(is_dict_normalized_);
  CHECK_EQ(conv_out_spatial_dim_, 1) << "Convolutional dictionaries not implemented, yet!";
  // Temporary storage and precomputed constants
  int m = kernel_dim_;
  int k = num_output_;
  Dtype* tmp1 = tmp_buffer_.mutable_gpu_data();
  Dtype* tmp2 = tmp_buffer_.mutable_gpu_data() + std::max(k,m);
  Dtype* tmp3 = tmp_buffer_.mutable_gpu_data() + 2*std::max(k,m);
  Dtype* tmp_dl_dx = tmp_buffer_.mutable_gpu_data() + 3*std::max(k,m);
  // Precomputed matrices
  const Dtype* Vt = this->blobs_[bias_idx_ + 2]->gpu_data();
  const Dtype* Vt_sn2 = Vt_sn2_buffer_.gpu_data();
  const Dtype* Ddagger = Ddagger_buffer_.gpu_data();
  for (int idx = 0; idx < top.size()/2; ++idx) {
    const Dtype* top_diff = top[2*idx]->gpu_diff();
    const Dtype* top_data = top[2*idx]->gpu_data();
    const Dtype* bottom_data = bottom[idx]->gpu_data();
    Dtype* bottom_diff = bottom[idx]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[bias_idx_]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[idx*2]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[idx]) {
      for (int n = 0; n < this->num_; ++n) {
        const Dtype* alpha = top_data + top[idx*2]->offset(n);
        const Dtype* alpha_diff = top_diff + top[idx*2]->offset(n);
        // Precompute modified output gradient
        Dtype* mod_alpha_diff = mod_alpha_diff_buffer_.mutable_gpu_data();
        kernel_mod_gradient<Dtype><<<CAFFE_GET_BLOCKS(k),
            CAFFE_CUDA_NUM_THREADS>>>(k, alpha, alpha_diff, mod_alpha_diff);
        // dl/dx is necessary for both gradients
        Dtype* dl_dx = propagate_down[idx] ?
              bottom_diff + bottom[idx]->offset(n) : tmp_dl_dx;
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[idx] || this->param_propagate_down_[0]) {
          this->backward_gpu_gemm(mod_alpha_diff, Ddagger, dl_dx);
        }
        // gradient w.r.t. dictionary. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->dict_gpu_backprop(bottom_data + bottom[idx]->offset(n),
              dl_dx, mod_alpha_diff, Ddagger, Vt, Vt_sn2, tmp1, tmp2, tmp3,
              D_diff);
          this->dict_gpu_optimize(bottom_data + bottom[idx]->offset(n), alpha,
              D, (Dtype)etha_, tmp1, tmp2, D_diff);
          // Mark dictionary as unnormalized
          is_dict_normalized_ = false;
          Dtype* Dflag = this->blobs_[bias_idx_ + 1]->mutable_cpu_data();
          *Dflag = (Dtype)1.;
        }
      }
    }
  }
}

template <typename Dtype>
void DictionaryLayer<Dtype>::dict_gpu_backprop(const Dtype* x, const Dtype* dl_dx,
      const Dtype* mod_alpha_diff, const Dtype* Dtdagger, const Dtype* Vt,
      const Dtype* Vt_sn2, Dtype* tmp1, Dtype* tmp2, Dtype* tmp3, Dtype* D_diff) {
  int m = kernel_dim_;
  int k = num_output_;
  int r = rank_;
  // Compute intermediate products
  // tmp1 = x^T * (D^dagger)^T
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, k, m, (Dtype)1., x,
      Dtdagger, (Dtype)0., tmp1);
  // tmp2 = dl_dalpha * V
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, r, k, (Dtype)1.,
      mod_alpha_diff, Vt, (Dtype)0., tmp2);
  // tmp3 = tmp2 * Vt_sn2
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, k, r, (Dtype)1.,
      tmp2, Vt_sn2, (Dtype)0., tmp3);
  // Compute gradient of dictionary and add it to D_diff
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, k, 1, -(Dtype)2., dl_dx,
      tmp1, (Dtype)1., D_diff);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, k, 1, (Dtype)1., x,
      tmp3, (Dtype)1., D_diff);
  // Result to be D_diff = D_diff - 2 * (D^dagger)^T * x^T * mod_alpha_diff^t * (D^dagger)^T
  //                              + x * mod_alpha_diff * V * Vt_sn2

}

template <typename Dtype>
void DictionaryLayer<Dtype>::dict_gpu_optimize(const Dtype* x,
    const Dtype* alpha, const Dtype* D, Dtype etha, Dtype* tmp1, Dtype* tmp2,
    Dtype* D_diff) {
  if (etha == (Dtype)0.)
    return;
  int m = kernel_dim_;
  int k = num_output_;
  caffe_copy(m, x, tmp1);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, 1, k,
      (Dtype)1., D, alpha, -(Dtype)1., tmp1);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, k, 1,
      etha, tmp1, alpha, (Dtype)1., D_diff);
}

template <typename Dtype>
void DictionaryLayer<Dtype>::backward_gpu_gemm(const Dtype* mod_alpha_diff,
    const Dtype* D, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, kernel_dim_,
      1, num_output_,
      (Dtype)1., D, mod_alpha_diff,
      (Dtype)0., col_buff);
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void DictionaryLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  int k = num_output_;
  caffe_gpu_add(k, input, bias, bias);
}

INSTANTIATE_LAYER_GPU_FUNCS(DictionaryLayer);

}  // namespace caffe
