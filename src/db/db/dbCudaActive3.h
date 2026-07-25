/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaActive3
#define HDR_dbCudaActive3

#include "dbCommon.h"
#include "dbEdgePairRelations.h"
#include "dbRegionLocalOperations.h"

namespace db
{

class DeepLayer;

/**
 * Try the narrowly qualified live ACTIVE.3 empty certificate.
 *
 * False means "run the pristine CPU implementation."  True is returned only
 * for a fully echoed COMPLETE backend result with no raw hit, uncertainty,
 * fallback flag, or device flag.
 */
DB_PUBLIC bool cuda_active3_try_empty (
  db::edge_relation_type relation, bool different_polygons, db::Coord distance,
  const db::RegionCheckOptions &options, const db::DeepLayer &merged_well,
  const db::DeepLayer &raw_active);

/**
 * Try ACTIVE.3 before constructing merged WELL or merged ACTIVE.
 *
 * The indexed operand is the complete raw NWELL contour stream followed by
 * the complete raw PWELL contour stream.  The primary is complete raw ACTIVE.
 * True is returned only for an exact, fully echoed zero-hit certificate.
 * Every decline, hit, uncertainty, error, or malformed input means that the
 * caller must retain the historical WELL union and ACTIVE.3 implementation.
 */
DB_PUBLIC bool cuda_active3_raw_wells_try_empty (
  const db::DeepLayer &raw_nwell, const db::DeepLayer &raw_pwell,
  const db::DeepLayer &raw_active);

/**
 * Try exact raw-(NWELL union PWELL) followed by ACTIVE.3 on one device.
 *
 * Unlike the retained raw-WELL superset experiment, this path forms the exact
 * WELL integer set before applying the exact ACTIVE.3 predicate to complete
 * raw ACTIVE.  True is returned only for a fully echoed, zero-hit,
 * zero-uncertainty resident certificate.  False preserves the historical
 * WELL union and CPU rule.
 */
DB_PUBLIC bool cuda_active3_well_union_try_empty (
  const db::DeepLayer &raw_nwell, const db::DeepLayer &raw_pwell,
  const db::DeepLayer &raw_active);

/**
 * Try the narrowly qualified live CONTACT.4 empty certificate.
 *
 * The primary is merged FreePDK45 GDS layer 1/0 (ACTIVE) and the secondary is
 * the raw layer 10/0 (CONTACT) superset.  False always means "run the pristine
 * CPU implementation."  True is returned only for a fully echoed COMPLETE
 * backend result with no raw hit, uncertainty, fallback flag, or device flag.
 */
DB_PUBLIC bool cuda_contact4_try_empty (
  db::edge_relation_type relation, bool different_polygons, db::Coord distance,
  const db::RegionCheckOptions &options, const db::DeepLayer &merged_active,
  const db::DeepLayer &raw_active, const db::DeepLayer &raw_contact);

/**
 * Try CONTACT.4 before constructing merged ACTIVE.
 *
 * This opt-in indexes the complete raw CONTACT secondary and streams the
 * complete raw ACTIVE primary.  A certified zero-hit raw superset permits an
 * empty return; every other outcome leaves the established merged-CUDA/CPU
 * path untouched.
 */
DB_PUBLIC bool cuda_contact4_raw_active_try_empty (
  db::edge_relation_type relation, bool different_polygons, db::Coord distance,
  const db::RegionCheckOptions &options, const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_contact);

/**
 * Try exact raw-ACTIVE union followed by CONTACT.4 on one resident device.
 *
 * This opt-in runs before constructing merged ACTIVE.  It serializes the two
 * complete raw physical layers into separate, digest-bound compact scenes
 * which share one hierarchy identity.  True is returned only when the exact
 * ACTIVE integer-set union and complete CONTACT.4 scan both finish on the
 * selected device with zero hits.  False preserves the established fallback.
 */
DB_PUBLIC bool cuda_contact4_active_union_try_empty (
  db::edge_relation_type relation, bool different_polygons, db::Coord distance,
  const db::RegionCheckOptions &options, const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_contact);

} // namespace db

#endif
