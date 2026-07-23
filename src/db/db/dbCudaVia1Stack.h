/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaVia1Stack
#define HDR_dbCudaVia1Stack

#include "dbCommon.h"

namespace db
{

class DeepLayer;

/**
 * Try the qualified atomic M1/VIA1/M2 CUDA empty certificate.
 *
 * True certifies all six fixed FreePDK45 rules.  False is a normal decline and
 * requires the caller to execute the complete historical CPU rule stack.
 */
DB_PUBLIC bool cuda_via1_stack_try_empty (
  const db::DeepLayer &raw_metal1, const db::DeepLayer &raw_via1,
  const db::DeepLayer &raw_metal2);

} // namespace db

#endif
