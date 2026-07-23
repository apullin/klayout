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

} // namespace db

#endif
