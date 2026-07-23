/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaActive3Digest
#define HDR_dbCudaActive3Digest

#include "dbCudaSpatialApi.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace db
{
namespace cuda_active3_digest
{

class Sha256
{
public:
  Sha256 ()
    : m_state {{ 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u }},
      m_block (), m_total_bytes (0), m_block_bytes (0)
  {
    //  nothing yet
  }

  void update (const void *data, std::size_t bytes)
  {
    const std::uint8_t *source = static_cast<const std::uint8_t *> (data);
    m_total_bytes += bytes;
    while (bytes) {
      const std::size_t available = m_block.size () - m_block_bytes;
      const std::size_t take = std::min (available, bytes);
      std::memcpy (m_block.data () + m_block_bytes, source, take);
      m_block_bytes += take;
      source += take;
      bytes -= take;
      if (m_block_bytes == m_block.size ()) {
        transform (m_block.data ());
        m_block_bytes = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finish ()
  {
    const std::uint64_t bit_count = m_total_bytes * 8;
    const std::uint8_t marker = 0x80;
    update (&marker, 1);
    const std::uint8_t zero = 0;
    while (m_block_bytes != 56) {
      update (&zero, 1);
    }
    std::uint8_t length [8];
    for (int i = 0; i < 8; ++i) {
      length [7 - i] = static_cast<std::uint8_t> (bit_count >> (i * 8));
    }
    update (length, sizeof (length));

    std::array<std::uint8_t, 32> digest;
    for (int i = 0; i < 8; ++i) {
      digest [i * 4] = static_cast<std::uint8_t> (m_state [i] >> 24);
      digest [i * 4 + 1] = static_cast<std::uint8_t> (m_state [i] >> 16);
      digest [i * 4 + 2] = static_cast<std::uint8_t> (m_state [i] >> 8);
      digest [i * 4 + 3] = static_cast<std::uint8_t> (m_state [i]);
    }
    return digest;
  }

private:
  static std::uint32_t rotate_right (std::uint32_t value, unsigned int amount)
  {
    return (value >> amount) | (value << (32 - amount));
  }

  void transform (const std::uint8_t *input)
  {
    static const std::uint32_t constants [64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
      0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
      0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
      0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };

    std::uint32_t words [64];
    for (int i = 0; i < 16; ++i) {
      words [i] =
        (static_cast<std::uint32_t> (input [i * 4]) << 24) |
        (static_cast<std::uint32_t> (input [i * 4 + 1]) << 16) |
        (static_cast<std::uint32_t> (input [i * 4 + 2]) << 8) |
        static_cast<std::uint32_t> (input [i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 =
        rotate_right (words [i - 15], 7) ^
        rotate_right (words [i - 15], 18) ^ (words [i - 15] >> 3);
      const std::uint32_t s1 =
        rotate_right (words [i - 2], 17) ^
        rotate_right (words [i - 2], 19) ^ (words [i - 2] >> 10);
      words [i] = words [i - 16] + s0 + words [i - 7] + s1;
    }

    std::uint32_t a = m_state [0], b = m_state [1];
    std::uint32_t c = m_state [2], d = m_state [3];
    std::uint32_t e = m_state [4], f = m_state [5];
    std::uint32_t g = m_state [6], h = m_state [7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t sum1 =
        rotate_right (e, 6) ^ rotate_right (e, 11) ^ rotate_right (e, 25);
      const std::uint32_t choose = (e & f) ^ (~e & g);
      const std::uint32_t temp1 =
        h + sum1 + choose + constants [i] + words [i];
      const std::uint32_t sum0 =
        rotate_right (a, 2) ^ rotate_right (a, 13) ^ rotate_right (a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    m_state [0] += a;
    m_state [1] += b;
    m_state [2] += c;
    m_state [3] += d;
    m_state [4] += e;
    m_state [5] += f;
    m_state [6] += g;
    m_state [7] += h;
  }

  std::array<std::uint32_t, 8> m_state;
  std::array<std::uint8_t, 64> m_block;
  std::uint64_t m_total_bytes;
  std::size_t m_block_bytes;
};

inline bool checked_bytes (std::uint64_t count, std::size_t record_size,
                           std::size_t &bytes)
{
  if (count > std::numeric_limits<std::size_t>::max () / record_size) {
    return false;
  }
  bytes = static_cast<std::size_t> (count) * record_size;
  return true;
}

inline bool request_digest (
  const klayout_cuda_spatial_active3_request_v1 &request,
  std::array<std::uint8_t, 32> &digest)
{
  if ((request.context_count && ! request.contexts) ||
      (request.well_context_count && ! request.well_contexts) ||
      (request.well_offset_count && ! request.well_offsets) ||
      (request.active_context_count && ! request.active_contexts) ||
      (request.cell_count && ! request.cells) ||
      (request.edge_count && ! request.edges)) {
    return false;
  }

  static const char magic [8] = { 'K', 'A', 'C', 'T', 'L', 'I', 'V', 'E' };
  Sha256 sha;
  sha.update (magic, sizeof (magic));
#define KLAYOUT_ACTIVE3_DIGEST_FIELD(field) \
  sha.update (&request.field, sizeof (request.field))
  KLAYOUT_ACTIVE3_DIGEST_FIELD (abi_version);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (struct_size);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (opcode);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (option_flags);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (dbu_per_micron);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (distance);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (grid_cell_size);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (context_count);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (well_context_count);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (well_offset_count);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (active_context_count);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (cell_count);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (edge_count);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (flat_well_edge_count);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (flat_active_edge_count);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (well_left);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (well_bottom);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (well_right);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (well_top);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (max_contexts);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (max_grid_cells);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (max_memberships);
  KLAYOUT_ACTIVE3_DIGEST_FIELD (max_pair_work);
#undef KLAYOUT_ACTIVE3_DIGEST_FIELD

  std::size_t bytes = 0;
#define KLAYOUT_ACTIVE3_DIGEST_ARRAY(pointer, count, type) \
  do { \
    if (! checked_bytes (request.count, sizeof (type), bytes)) return false; \
    if (bytes) sha.update (request.pointer, bytes); \
  } while (false)
  KLAYOUT_ACTIVE3_DIGEST_ARRAY (
    contexts, context_count, klayout_cuda_spatial_active3_context_v1);
  KLAYOUT_ACTIVE3_DIGEST_ARRAY (well_contexts, well_context_count, uint32_t);
  KLAYOUT_ACTIVE3_DIGEST_ARRAY (well_offsets, well_offset_count, uint64_t);
  KLAYOUT_ACTIVE3_DIGEST_ARRAY (
    active_contexts, active_context_count, uint32_t);
  KLAYOUT_ACTIVE3_DIGEST_ARRAY (
    cells, cell_count, klayout_cuda_spatial_active3_cell_v1);
  KLAYOUT_ACTIVE3_DIGEST_ARRAY (
    edges, edge_count, klayout_cuda_spatial_active3_edge_v1);
#undef KLAYOUT_ACTIVE3_DIGEST_ARRAY
  digest = sha.finish ();
  return true;
}

} // namespace cuda_active3_digest
} // namespace db

#endif
