/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaImplant12
#define HDR_dbCudaImplant12

#include "dbCommon.h"
#include "dbCudaSpatialApi.h"

#include <cstdint>
#include <string>
#include <vector>

namespace db
{

class DeepLayer;

struct DB_PUBLIC CudaImplant12SceneLimits
{
  uint64_t max_contexts;
  uint64_t max_flat_polygons;
  uint64_t max_flat_contours;
  uint64_t max_flat_edges;

  CudaImplant12SceneLimits ();
};

/**
 * Region-state assertions supplied by the GSI transaction boundary.
 *
 * DeepLayer retains geometry and hierarchy, but not the Region delegate's
 * merged/raw state.  The builder therefore requires that state explicitly.
 */
struct DB_PUBLIC CudaImplant12BuildSpec
{
  bool implant_is_exact_merged;
  bool gate_is_raw;
  bool contact_is_raw;

  CudaImplant12BuildSpec ();
};

/** Owning, pointer-free scene used to populate the optional backend ABI. */
struct DB_PUBLIC CudaImplant12Scene
{
  std::vector<klayout_cuda_spatial_implant12_context_v1> contexts;
  std::vector<uint32_t> implant_contexts;
  std::vector<uint64_t> implant_edge_offsets;
  std::vector<uint32_t> gate_contexts;
  std::vector<uint32_t> contact_contexts;
  std::vector<klayout_cuda_spatial_implant12_cell_v1> cells;
  std::vector<klayout_cuda_spatial_implant12_contour_v1> contours;
  std::vector<klayout_cuda_spatial_implant12_edge_v1> edges;
  uint32_t root_cell;
  uint64_t flat_polygons [KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT];
  uint64_t flat_contours [KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT];
  uint64_t flat_edges [KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT];
  bool have_implant_bounds;
  int64_t implant_left;
  int64_t implant_bottom;
  int64_t implant_right;
  int64_t implant_top;

  CudaImplant12Scene ();
  void swap (CudaImplant12Scene &other) noexcept;
};

/**
 * Build the narrowly qualified three-domain hierarchy scene.
 *
 * On every unsupported shape, hierarchy, provenance, DBU, overflow or
 * capacity condition this returns false and leaves scene unchanged.
 */
DB_PUBLIC bool cuda_implant12_build_scene (
  const db::DeepLayer &merged_implant, const db::DeepLayer &raw_gate,
  const db::DeepLayer &raw_contact, const CudaImplant12BuildSpec &spec,
  const CudaImplant12SceneLimits &limits, CudaImplant12Scene &scene,
  std::string *decline_reason = 0);

/**
 * Try the qualified atomic FreePDK45 IMPLANT.1/IMPLANT.2 empty certificate.
 *
 * The primary is the exact merged NPLUS-or-PPLUS IMPLANT result.  GATE and
 * CONTACT are the raw relation operands.  True certifies both ordered rules
 * empty.  False is a normal fail-closed decline and requires the caller to
 * execute both historical CPU expressions.
 */
DB_PUBLIC bool cuda_implant12_try_empty (
  const db::DeepLayer &merged_implant, const db::DeepLayer &raw_gate,
  const db::DeepLayer &raw_contact);

} // namespace db

#endif
