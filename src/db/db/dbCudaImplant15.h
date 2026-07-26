/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaImplant15
#define HDR_dbCudaImplant15

#include "dbCommon.h"

namespace db
{

class DeepLayer;

/**
 * Try the exact raw FreePDK45 IMPLANT.1-.5 resident transaction.
 *
 * NPLUS, PPLUS and CONTACT are pristine physical layers.  GATE is the exact
 * already-derived POLY-and-ACTIVE DeepLayer and deliberately carries no
 * physical-layer claim.  True means all five historical output universes are
 * certified empty by one fully echoed device transaction.  False is a normal
 * fail-closed decline and requires all five literal CPU expressions.
 */
DB_PUBLIC bool cuda_implant15_try_raw_empty (
  const db::DeepLayer &raw_nplus,
  const db::DeepLayer &raw_pplus,
  const db::DeepLayer &derived_gate,
  const db::DeepLayer &raw_contact);

} // namespace db

#endif
