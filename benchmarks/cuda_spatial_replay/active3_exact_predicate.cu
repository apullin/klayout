/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "active3_exact_predicate.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace klayout_cuda {
namespace active3 {
namespace {

__global__ void classify_kernel(const EdgePair *pairs, std::size_t count,
                                std::int64_t distance, Verdict *results) {
  std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  while (index < count) {
    results[index] = classify_pair_bounded(pairs[index], distance);
    if (count - index <= stride) {
      break;
    }
    index += stride;
  }
}

bool cuda_ok(cudaError_t status, const char *operation, std::string *error) {
  if (status == cudaSuccess) {
    return true;
  }
  if (error) {
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    *error = message.str();
  }
  return false;
}

}  // namespace

bool classify_batch(const EdgePair *pairs, std::size_t count,
                    std::int64_t distance, Verdict *results,
                    std::string *error) {
  if (error) {
    error->clear();
  }
  if (count == 0) {
    return true;
  }
  if (!pairs || !results) {
    if (error) {
      *error = "null batch input or output";
    }
    return false;
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(EdgePair) ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(Verdict)) {
    if (error) {
      *error = "batch byte-size overflow";
    }
    return false;
  }

  EdgePair *device_pairs = nullptr;
  Verdict *device_results = nullptr;
  const std::size_t pair_bytes = count * sizeof(EdgePair);
  const std::size_t result_bytes = count * sizeof(Verdict);

  if (!cuda_ok(cudaMalloc(&device_pairs, pair_bytes), "cudaMalloc(pairs)",
               error)) {
    return false;
  }
  if (!cuda_ok(cudaMalloc(&device_results, result_bytes),
               "cudaMalloc(results)", error)) {
    cudaFree(device_pairs);
    return false;
  }

  bool ok = cuda_ok(cudaMemcpy(device_pairs, pairs, pair_bytes,
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(pairs H2D)", error);
  if (ok) {
    constexpr unsigned int kBlockSize = 256;
    constexpr std::size_t kPortableMaximumGridX = 65535;
    const std::size_t needed_blocks = (count - 1) / kBlockSize + 1;
    // 65,535 is supported even by legacy CUDA devices.  A grid-stride loop
    // handles larger batches without truncating size_t into the launch type.
    const unsigned int blocks = static_cast<unsigned int>(
        needed_blocks < kPortableMaximumGridX ? needed_blocks
                                               : kPortableMaximumGridX);
    classify_kernel<<<blocks, kBlockSize>>>(device_pairs, count, distance,
                                            device_results);
    ok = cuda_ok(cudaGetLastError(), "classify_kernel launch", error);
  }
  if (ok) {
    ok = cuda_ok(cudaMemcpy(results, device_results, result_bytes,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(results D2H)", error);
  }

  const cudaError_t free_results = cudaFree(device_results);
  const cudaError_t free_pairs = cudaFree(device_pairs);
  if (ok && !cuda_ok(free_results, "cudaFree(results)", error)) {
    ok = false;
  }
  if (ok && !cuda_ok(free_pairs, "cudaFree(pairs)", error)) {
    ok = false;
  }
  return ok;
}

const char *verdict_name(Verdict verdict) {
  switch (verdict) {
    case Verdict::kNoViolation:
      return "NO_VIOLATION";
    case Verdict::kViolation:
      return "VIOLATION";
    case Verdict::kUncertain:
      return "UNCERTAIN";
  }
  return "INVALID";
}

}  // namespace active3
}  // namespace klayout_cuda
