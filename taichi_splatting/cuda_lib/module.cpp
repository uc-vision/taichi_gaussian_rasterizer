#include <pybind11/pybind11.h>
#include <torch/extension.h>


int64_t full_cumsum(const torch::Tensor input, const torch::Tensor output);

std::pair<torch::Tensor, torch::Tensor> segmented_sort_pairs(
  const torch::Tensor keys, const torch::Tensor values,
  const torch::Tensor start_offset, const torch::Tensor end_offset);

std::pair<torch::Tensor, torch::Tensor> segmented_radix_sort(
  const torch::Tensor keys, const torch::Tensor values,
  const torch::Tensor start_offset, const torch::Tensor end_offset,
  int begin_bit=0, int end_bit=-1);


std::pair<torch::Tensor, torch::Tensor> radix_sort_pairs(
  const torch::Tensor keys, const torch::Tensor values, int begin_bit=0, int end_bit=-1, bool force_unsigned=false);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("full_cumsum", &full_cumsum, "Full cumulative sum");
  m.def("segmented_sort_pairs", &segmented_sort_pairs, "Segmented sort pairs");
  m.def("segmented_radix_sort", &segmented_radix_sort, "Segmented radix sort");
  
  m.def("radix_sort_pairs", &radix_sort_pairs, "Sort pairs by radix sort");
}