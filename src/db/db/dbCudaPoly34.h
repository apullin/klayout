/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaPoly34
#define HDR_dbCudaPoly34

#include "dbCommon.h"
#include "dbCudaSpatialApi.h"

#include <stdint.h>
#include <string>
#include <vector>

namespace db
{

class DeepLayer;

struct DB_PUBLIC CudaPoly34SceneLimits
{
  CudaPoly34SceneLimits ();

  uint64_t max_contexts;
  uint64_t max_flat_boxes;
};

struct DB_PUBLIC CudaPoly34BuildSpec
{
  CudaPoly34BuildSpec ();

  bool poly_is_exact_merged;
  bool active_is_exact_merged;
  bool gate_is_exact_merged;
  bool freepdk45_layer_contract;
};

struct DB_PUBLIC CudaPoly34Scene
{
  CudaPoly34Scene ();
  void swap (CudaPoly34Scene &other) noexcept;

  std::vector<klayout_cuda_spatial_poly34_context_v1> contexts;
  std::vector<uint32_t> poly_contexts;
  std::vector<uint64_t> poly_offsets;
  std::vector<uint32_t> active_contexts;
  std::vector<uint64_t> active_offsets;
  std::vector<uint32_t> gate_contexts;
  std::vector<uint64_t> gate_offsets;
  std::vector<klayout_cuda_spatial_poly34_cell_v1> cells;
  std::vector<klayout_cuda_spatial_poly34_box_v1> boxes;
  uint32_t root_cell;
  uint64_t flat_boxes[KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT];
  bool have_scene_box;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;
  uint64_t store_identity;
  uint64_t layout_identity;
  uint64_t top_cell_identity;
  uint32_t layer_ids[KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT];
};

/**
 * Build the compact hierarchical POLY.3/.4 box scene without flattening.
 *
 * False is a normal fail-closed decline.  The destination is changed only
 * after the complete scene and all checked censuses have been validated.
 */
DB_PUBLIC bool cuda_poly34_build_scene (
  const db::DeepLayer &merged_poly, const db::DeepLayer &merged_active,
  const db::DeepLayer &merged_gate, const CudaPoly34BuildSpec &spec,
  const CudaPoly34SceneLimits &limits, CudaPoly34Scene &scene,
  std::string *decline_reason = 0);

/**
 * Build format 2 from pristine physical POLY and ACTIVE without constructing
 * either merged operand or the derived GATE layer on the host.
 *
 * The two domains are exact rectangle covers.  The GATE domain is represented
 * by empty spans and lists and KLAYOUT_CUDA_SPATIAL_POLY34_NO_GATE_LAYER.
 */
DB_PUBLIC bool cuda_poly34_build_raw_scene (
  const db::DeepLayer &raw_poly, const db::DeepLayer &raw_active,
  const CudaPoly34SceneLimits &limits, CudaPoly34Scene &scene,
  std::string *decline_reason = 0);

/**
 * Try the optional atomic POLY.3/POLY.4 terminal-empty certificate.
 *
 * True certifies both historical output categories empty.  False means run
 * both pristine CPU expressions; it is returned for every hit, uncertainty,
 * capacity, malformed request, missing backend or CUDA failure.
 */
DB_PUBLIC bool cuda_poly34_try_empty (
  const db::DeepLayer &merged_poly, const db::DeepLayer &merged_active,
  const db::DeepLayer &merged_gate, const CudaPoly34BuildSpec &spec);

/**
 * Try format 2 directly from pristine physical POLY and ACTIVE.
 *
 * True is consumable only when the backend derived the complete intersection
 * and certified both terminal output categories empty.  False requires the
 * historical host GATE construction and both CPU expressions.
 */
DB_PUBLIC bool cuda_poly34_try_raw_empty (
  const db::DeepLayer &raw_poly, const db::DeepLayer &raw_active);

} // namespace db

#endif
