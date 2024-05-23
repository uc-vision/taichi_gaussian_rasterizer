#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cub/cub.cuh>


// for our workload segmented sort (non radix) seems faster
template <typename K, typename V>
void sort_helper(
  K *d_keys, V *d_values, 
  K *d_keys_out, V *d_values_out,

  int num_items, 
  int64_t *d_start_offset, int64_t *d_end_offset, 
  int num_segments,   int begin_bit=0, int end_bit=-1)  
{

  size_t   temp_storage_bytes = 0;
  end_bit = end_bit > 0 ? end_bit : sizeof(K) * 8;
  auto stream = at::cuda::getCurrentCUDAStream();

  cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes,
      d_keys, d_keys_out, d_values, d_values_out,
      num_items, num_segments, 
      d_start_offset, d_end_offset, 
      begin_bit, end_bit, stream);

  auto temp_storage = torch::empty({int64_t(temp_storage_bytes)}, 
    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

  cub::DeviceSegmentedRadixSort::SortPairs(temp_storage.data_ptr<uint8_t>(), temp_storage_bytes,
      d_keys, d_keys_out, d_values, d_values_out,
      num_items, num_segments, 
      d_start_offset, d_end_offset, 
      begin_bit, end_bit, stream);

}


std::pair<torch::Tensor, torch::Tensor> segmented_radix_sort(
  const torch::Tensor keys, const torch::Tensor values,
  const torch::Tensor start_offset, const torch::Tensor end_offset,
  int begin_bit=0, int end_bit=-1) {
  
  assert (keys.dim() == 1 && values.dim() == 1), "keys and values must be 1D";
  assert (keys.size(0) == values.size(0)), "keys and values must have the same size";

  assert (start_offset.dim() == 1 && end_offset.dim() == 1 && start_offset.size(0) == end_offset.size(0)), 
    "start_offset and end_offset must be 1D and have the same size"; 

  assert (start_offset.scalar_type() == torch::kInt64 
    && end_offset.scalar_type() == torch::kInt64), "start_offset/end_offset must be int64";
  
  auto keys_out = torch::empty_like(keys);
  auto values_out = torch::empty_like(values);

  if (keys.scalar_type() == torch::kInt32 && values.scalar_type() == torch::kInt32) {
    sort_helper<int32_t, int32_t>(
      keys.data_ptr<int32_t>(), values.data_ptr<int32_t>(), 
      keys_out.data_ptr<int32_t>(), values_out.data_ptr<int32_t>(),
      keys.size(0), 
      start_offset.data_ptr<int64_t>(), end_offset.data_ptr<int64_t>(),
      start_offset.size(0), begin_bit, end_bit);

      return std::make_pair(keys_out, values_out);
  } else if (keys.scalar_type() == torch::kInt16 && values.scalar_type() == torch::kInt32) {
    sort_helper<int16_t, int32_t>(
      keys.data_ptr<int16_t>(), values.data_ptr<int32_t>(), 
      keys_out.data_ptr<int16_t>(), values_out.data_ptr<int32_t>(),
      keys.size(0), 
      start_offset.data_ptr<int64_t>(), end_offset.data_ptr<int64_t>(),
      start_offset.size(0), begin_bit, end_bit);

      return std::make_pair(keys_out, values_out);
  } else { 
      // TODO, add all the other cases.
      throw std::runtime_error("Not yet implemented for data type.");
  }
}

