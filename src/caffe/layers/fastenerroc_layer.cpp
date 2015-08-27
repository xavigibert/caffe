#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <limits>
#include <iomanip>      // std::setprecision

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FastenerRocLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  num_classes_ = this->layer_param_.fastenerroc_param().num_classes();
  num_classes_ext_ = this->layer_param_.fastenerroc_param().num_classes_ext();
  num_good_ = this->layer_param_.fastenerroc_param().num_good();
  eof_marker_ = this->layer_param_.fastenerroc_param().eof_marker();
  desired_pfa_ = this->layer_param_.fastenerroc_param().desired_pfa();
  computed_pd_ = Dtype(0);
  auc_ = Dtype(0);
}

template <typename Dtype>
void FastenerRocLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_shape);
}

// Computes ROC curve from a list of annotated scores (sorts scores)
static void computeRocCurve(std::vector< std::pair<double, int> >& scores,
             std::vector<double>& vpd, std::vector<double>& vpf, double& auc)
{
    std::sort(scores.begin(), scores.end());
    auc = 0.0;
    double last_pd = 0.0;
    double last_pf = 0.0;
    int n = scores.size();
    vpd.resize(n);
    vpf.resize(n);
    // Count number of positives and negatives
    int pos = 0;
    int neg = 0;
    for( int i = 0; i < n; i++ )
    {
        if( scores[i].second > 0 )
            pos++;
    }
    neg = n - pos;
    if( pos == 0 || neg == 0 )
        return;
    // Calculate pfa and pd, and add the area under the curve
    int tp = 0;
    int fp = 0;
    for( int i = 0; i < n; i++ )
    {
        if( scores[i].second > 0 )
            tp++;
        else
            fp++;
        double pd = double(tp) / double(pos);
        double pf = double(fp) / double(neg);
        if( pf != last_pf )
            auc += (pf - last_pf) * (pd + last_pd) / 2.0;
        vpd[i] = last_pd = pd;
        vpf[i] = last_pf = pf;
    }
    // Last trapezoid
    if( 1.0 != last_pf )
        auc += (1.0 - last_pf) * (1.0 + last_pd) / 2.0;
}

// Compute (interpolated) constant false alarm rate (CFAR) point from an ROC curve
// PREREQUISITE: vpf should be in ascending order
double computeCfarTh(const std::vector<std::pair<double, int> >& scores, const std::vector<double>& vpd, const std::vector<double>& vpf, double pfa, double& pd)
{
    int n = vpf.size();
    CHECK_EQ(vpf.size(), vpd.size());
    if (vpf[0] >= pfa)
      return 0;
    for( int i = 1; i < n; i++ )
    {
        if ( vpf[i] >= pfa )
        {
            double alpha =  (vpf[i] - pfa) / (vpf[i] - vpf[i-1]);
            pd = (1-alpha) * vpd[i] + alpha * vpd[i-1];
            return (1-alpha) * scores[i].first + alpha * scores[i-1].first;
        }
    }
    return scores[n-1].first;
}

template <typename Dtype>
void FastenerRocLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2*num_classes_ext_+1);
  std::vector<const Dtype*> bottom_data(2*num_classes_ext_);
  int batch_size = bottom[0]->shape(0);
  for (int i = 0; i < 2*num_classes_ext_; ++i) {
    bottom_data[i] = bottom[i]->cpu_data();
    CHECK_EQ(bottom[i]->count(), 2*batch_size);
  }
  const Dtype* bottom_label = bottom[2*num_classes_ext_]->cpu_data();

  for (int n = 0; n < batch_size; ++n) {
    int label_value = static_cast<int>(bottom_label[n]);
    double maxScoreGood = -std::numeric_limits<double>::max();
    double maxScoreBroken = -std::numeric_limits<double>::max();
    if (label_value != eof_marker_) {
      CHECK_GE(label_value, -1);
      CHECK_LT(label_value, num_classes_);
      // Convention:  1 (positive) = missing or broken fastneer
      //             -1 (negative) = good fastener
      label_value = (label_value < 0 || (label_value >= num_good_ && label_value < num_classes_)) ?
            1 : -1;
      for (int cl = 0; cl < num_classes_ext_; ++cl) {
        double fastVsBg = bottom_data[cl][2*n+1] - bottom_data[cl][2*n];
        double fastVsRest = bottom_data[cl+num_classes_ext_][2*n+1] - bottom_data[cl+num_classes_ext_][2*n];
        if (cl >= num_good_ && cl < num_classes_) {
          // Update broken score
          maxScoreBroken = std::max(maxScoreBroken, fastVsRest);
        }
        else {
          // Update missing score
          double penalty = std::min(fastVsRest, 0.);
          maxScoreGood = std::max(maxScoreGood, fastVsBg + penalty);
        }
      }
      double score = std::min(maxScoreGood,-maxScoreBroken);
      samples_.push_back(std::make_pair<double,int>(score,label_value));
    }
    else {
      // Generate ROC curve and calculate AUC
      std::vector<double> pd, pfa;
      computeRocCurve(samples_, pd, pfa, auc_);
      //LOG(INFO) << "AUC = " << auc;
      computeCfarTh(samples_, pd, pfa, desired_pfa_, computed_pd_);
      //LOG(INFO) << "PD(PFA=" << desired_pfa_ << ") = " << computed_pd_;
      // Save ROC curve
      std::ofstream f("roc_cnn.csv", std::ofstream::out);
      CHECK(f) << "Could not open roc_cnn.csv";
      for( int i = 0; i < samples_.size(); i++ )
      {
        f << std::setprecision(6) << pfa[i] << ","
          << pd[i] << ","
          << samples_[i].first << ","
          << samples_[i].second << std::endl;
      }
      samples_.clear();
    }
  }

  top[0]->mutable_cpu_data()[0] = static_cast<Dtype>(auc_);
  top[1]->mutable_cpu_data()[0] = static_cast<Dtype>(computed_pd_);
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(FastenerRocLayer);
REGISTER_LAYER_CLASS(FastenerRoc);

}  // namespace caffe
