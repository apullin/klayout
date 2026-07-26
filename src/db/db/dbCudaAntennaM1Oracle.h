/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaAntennaM1Oracle
#define HDR_dbCudaAntennaM1Oracle

#include "dbCommon.h"
#include "dbCudaAntennaM1.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace db
{

/**
 * Stable relation order used by the conductor connectivity digest.
 */
enum CudaAntennaM1OracleRelation
{
  CudaAntennaM1OraclePolySelf = 0,
  CudaAntennaM1OracleContactSelf = 1,
  CudaAntennaM1OracleMetal1Self = 2,
  CudaAntennaM1OraclePolyContact = 3,
  CudaAntennaM1OracleContactMetal1 = 4,
  CudaAntennaM1OracleRelationCount = 5
};

/**
 * Admission guards for the deterministic CPU M1 antenna graph oracle.
 *
 * This oracle is an explicitly requested integrity diagnostic, not a
 * production implementation.  The defaults admit focused fixtures and small
 * replays while declining a production-sized flattening before allocating it.
 */
struct DB_PUBLIC CudaAntennaM1OracleLimits
{
  uint64_t max_total_flat_polygons;
  uint64_t max_total_flat_edges;
  uint64_t max_graph_nodes;
  uint64_t max_candidate_pairs;

  CudaAntennaM1OracleLimits ();
};

/**
 * Canonical summary of the exact pre-diode M1 conductor prefix.
 *
 * Graph nodes are flattened POLY, CONTACT and M1 polygon occurrences.  Their
 * stable identity is (domain, context ID, source-cell index, polygon ID), with
 * the context parent/transform record bound into node_identity_digest.
 *
 * Connectivity consists only of the five production relations:
 *
 *   POLY self, CONTACT self, M1 self,
 *   POLY-CONTACT and CONTACT-M1.
 *
 * Each component is represented by its minimum node ID.  The annotation
 * digest additionally binds exact gate area (POLY & ACTIVE), exact merged M1
 * area and the factor-zero diode exemption boundary
 * NPLUS & (ACTIVE - NWELL) touching CONTACT.
 *
 * Diode is deliberately an annotation, not a graph node: one diode touching
 * multiple conductor roots marks every root exempt but does not union them.
 * That is outcome-equivalent for FreePDK45's factor-zero diode policy, while
 * keeping this certificate scoped to the reusable POLY/CONTACT/M1 GPU DSU.
 * These roots therefore are not final production-net roots in that corner.
 *
 * domain_component_counts is defined for POLY, CONTACT and M1.  Its three
 * annotation-domain entries (ACTIVE, NPLUS and NWELL) are always zero.
 */
struct DB_PUBLIC CudaAntennaM1Oracle
{
  uint32_t format_version;
  uint32_t reserved;
  uint64_t context_count;
  uint64_t graph_node_count;
  uint64_t component_count;
  uint64_t canonical_root_count;
  uint64_t top_context_component_count;
  uint64_t graph_candidate_pair_count;
  uint64_t graph_edge_count;
  uint64_t annotation_candidate_pair_count;
  //  FreePDK45 area-only antenna eligibility requires gate area > 1 DBU^2.
  uint64_t gate_component_count;
  uint64_t diode_exempt_component_count;
  uint64_t gate_area_dbu2;
  uint64_t metal1_area_dbu2;
  std::array<uint64_t, CudaAntennaM1DomainCount> domain_node_counts;
  std::array<uint64_t, CudaAntennaM1DomainCount> domain_component_counts;
  std::array<
    uint64_t, CudaAntennaM1OracleRelationCount>
    relation_candidate_pair_counts;
  std::array<
    uint64_t, CudaAntennaM1OracleRelationCount>
    relation_edge_counts;
  std::vector<uint64_t> canonical_labels;
  std::array<uint8_t, 32> node_identity_digest;
  //  GPU DSU certificate: identities, canonical roots and labels only.
  std::array<uint8_t, 32> partition_digest;
  //  Stricter certificate that also binds exact per-relation edge census.
  std::array<uint8_t, 32> connectivity_digest;
  std::array<uint8_t, 32> annotation_digest;
  std::array<uint8_t, 32> oracle_digest;

  CudaAntennaM1Oracle ();
};

/**
 * Build the deterministic host-only oracle from one validated capture.
 *
 * False is a normal fail-closed decline.  "oracle" is unchanged on every
 * failure.  No production DRC path calls this function.
 */
DB_PUBLIC bool cuda_antenna_m1_cpu_oracle (
  const CudaAntennaM1Capture &capture,
  const CudaAntennaM1OracleLimits &limits,
  CudaAntennaM1Oracle &oracle,
  std::string *decline_reason = 0);

/**
 * Render one stable, single-line oracle record for differential logs.
 */
DB_PUBLIC std::string cuda_antenna_m1_oracle_text (
  const CudaAntennaM1Oracle &oracle);

} // namespace db

#endif
