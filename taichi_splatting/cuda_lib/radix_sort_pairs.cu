#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cub/cub.cuh>


template <typename K, typename V>
void sort_helper(
  K *d_keys, V *d_values, 
  K *d_keys_out, V *d_values_out,
  int num_items,
  int begin_bit=0, int end_bit=-1) 
{
  size_t   temp_storage_bytes = 0;
  end_bit = end_bit > 0 ? end_bit : sizeof(K) * 8;
  auto stream = at::cuda::getCurrentCUDAStream();

  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
      d_keys, d_keys_out, d_values, d_values_out, num_items, begin_bit, end_bit, stream);

  auto temp_storage = torch::empty({int64_t(temp_storage_bytes)}, 
    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

  cub::DeviceRadixSort::SortPairs(temp_storage.data_ptr<uint8_t>(), temp_storage_bytes,
      d_keys, d_keys_out, d_values, d_values_out, num_items, begin_bit, end_bit, stream);

  cudaDeviceSynchronize();

}

std::pair<torch::Tensor, torch::Tensor> radix_sort_pairs(
  const torch::Tensor keys, const torch::Tensor values, int begin_bit=0, int end_bit=-1, bool force_unsigned=false) {
  
  assert (keys.dim() == 1 && values.dim() == 1), "keys and values must be 1D";
  assert (keys.size(0) == values.size(0)), "keys and values must have the same size";
  assert (values.scalar_type() == torch::kInt32), "values must be of type int32";


  auto keys_out = torch::empty_like(keys);
  auto values_out = torch::empty_like(values);

  if (keys.scalar_type() == torch::kInt32 && force_unsigned) {
    // hack as torch does not currently support unsigned integers

    sort_helper<uint32_t, int32_t>(
      (uint32_t*)keys.data_ptr<int32_t>(), values.data_ptr<int32_t>(), 
      (uint32_t*)keys_out.data_ptr<int32_t>(), values_out.data_ptr<int32_t>(),
      keys.size(0), begin_bit, end_bit);

      return std::make_pair(keys_out, values_out);

  } else if (keys.scalar_type() == torch::kInt16) {
    sort_helper<int16_t, int32_t>(
      keys.data_ptr<int16_t>(), values.data_ptr<int32_t>(), 
      keys_out.data_ptr<int16_t>(), values_out.data_ptr<int32_t>(),
      keys.size(0), begin_bit, end_bit);

      return std::make_pair(keys_out, values_out);

  } else if (keys.scalar_type() == torch::kInt32) {
    sort_helper<int32_t, int32_t>(
      keys.data_ptr<int32_t>(), values.data_ptr<int32_t>(), 
      keys_out.data_ptr<int32_t>(), values_out.data_ptr<int32_t>(),
      keys.size(0), begin_bit, end_bit);

      return std::make_pair(keys_out, values_out);

  } else if (keys.scalar_type() == torch::kInt64) {
    sort_helper<int64_t, int32_t>(
      keys.data_ptr<int64_t>(), values.data_ptr<int32_t>(), 
      keys_out.data_ptr<int64_t>(), values_out.data_ptr<int32_t>(),
      keys.size(0), begin_bit, end_bit);

      return std::make_pair(keys_out, values_out);

  } else { 
      // TODO, add all the other cases.
      throw std::runtime_error("Not yet implemented for data type(s).");
  }

  
}

