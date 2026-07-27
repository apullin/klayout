/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaAntennaM4Transaction
#define HDR_dbCudaAntennaM4Transaction

#include "dbCommon.h"
#include "dbDeepShapeStore.h"

namespace db
{

struct CudaAntennaM4Census;

/**
 * Exact field-wise comparison used by the optional prepared-census re-audit.
 */
DB_PUBLIC bool cuda_antenna_m1_m4_census_equal (
  const CudaAntennaM4Census &first,
  const CudaAntennaM4Census &second);

/**
 * Try the optional conservative ANTENNA.M1-through-M4 clean certificate.
 *
 * Capability discovery precedes provenance inspection and compact capture.
 * True means all four ordered 300:1 stages are fully certified clean by the
 * exact one-sided proof bound.  False is the normal outcome for every
 * disabled, unsupported, malformed, non-clean or exceptional path and keeps
 * the complete literal antenna deck mandatory.
 */
DB_PUBLIC bool cuda_antenna_m1_m4_try_raw_empty (
  const db::DeepLayer &raw_poly,
  const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_nplus,
  const db::DeepLayer &raw_nwell,
  const db::DeepLayer &raw_contact,
  const db::DeepLayer &raw_metal1,
  const db::DeepLayer &raw_via1,
  const db::DeepLayer &raw_metal2,
  const db::DeepLayer &raw_via2,
  const db::DeepLayer &raw_metal3,
  const db::DeepLayer &raw_via3,
  const db::DeepLayer &raw_metal4);

} // namespace db

#endif
