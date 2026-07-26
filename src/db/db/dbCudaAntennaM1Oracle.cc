/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaAntennaM1Oracle.h"

#include "dbCudaActive3Digest.h"
#include "dbPolygon.h"
#include "dbPolygonTools.h"
#include "dbRegion.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace db
{

namespace
{

const uint32_t oracle_format_version = 1;
const uint64_t default_max_total_flat_polygons = UINT64_C (24576);
const uint64_t default_max_total_flat_edges = UINT64_C (1048576);
const uint64_t default_max_graph_nodes = UINT64_C (8192);
const uint64_t default_max_candidate_pairs = UINT64_C (16777216);

const CudaAntennaM1Domain graph_domain_order [3] = {
  CudaAntennaM1Poly,
  CudaAntennaM1Contact,
  CudaAntennaM1Metal1
};

class AntennaM1OracleDecline
  : public std::runtime_error
{
public:
  explicit AntennaM1OracleDecline (const std::string &message)
    : std::runtime_error (message)
  {
    //  nothing yet
  }
};

class CanonicalDigest
{
public:
  void bytes (const void *data, size_t size)
  {
    m_sha.update (data, size);
  }

  void u32 (uint32_t value)
  {
    uint8_t encoded [4];
    for (unsigned int i = 0; i < 4; ++i) {
      encoded [i] = uint8_t (value >> (i * 8));
    }
    bytes (encoded, sizeof (encoded));
  }

  void u64 (uint64_t value)
  {
    uint8_t encoded [8];
    for (unsigned int i = 0; i < 8; ++i) {
      encoded [i] = uint8_t (value >> (i * 8));
    }
    bytes (encoded, sizeof (encoded));
  }

  void i64 (int64_t value)
  {
    u64 (static_cast<uint64_t> (value));
  }

  std::array<uint8_t, 32> finish ()
  {
    return m_sha.finish ();
  }

private:
  db::cuda_active3_digest::Sha256 m_sha;
};

struct OraclePolygon
{
  db::Polygon polygon;
  uint32_t domain;
  uint32_t context_id;
  uint32_t context_parent_id;
  uint32_t dense_cell_id;
  uint32_t polygon_id;
  uint64_t source_cell_index;
  uint64_t node_id;
};

class DeterministicDisjointSet
{
public:
  explicit DeterministicDisjointSet (size_t size)
    : m_parent (size)
  {
    for (size_t i = 0; i < size; ++i) {
      m_parent [i] = uint64_t (i);
    }
  }

  uint64_t find (uint64_t node)
  {
    uint64_t root = node;
    while (m_parent [size_t (root)] != root) {
      root = m_parent [size_t (root)];
    }
    while (m_parent [size_t (node)] != node) {
      const uint64_t next = m_parent [size_t (node)];
      m_parent [size_t (node)] = root;
      node = next;
    }
    return root;
  }

  void unite (uint64_t first, uint64_t second)
  {
    const uint64_t first_root = find (first);
    const uint64_t second_root = find (second);
    if (first_root == second_root) {
      return;
    }
    const uint64_t minimum = std::min (first_root, second_root);
    const uint64_t maximum = std::max (first_root, second_root);
    m_parent [size_t (maximum)] = minimum;
  }

private:
  std::vector<uint64_t> m_parent;
};

bool checked_add_u64 (uint64_t first, uint64_t second, uint64_t &result)
{
  if (second > std::numeric_limits<uint64_t>::max () - first) {
    return false;
  }
  result = first + second;
  return true;
}

uint64_t checked_multiply_u64 (
  uint64_t first, uint64_t second, const char *what)
{
  if (first && second >
      std::numeric_limits<uint64_t>::max () / first) {
    throw AntennaM1OracleDecline (
      std::string (what) + " overflows uint64");
  }
  return first * second;
}

uint64_t checked_choose_two (uint64_t count, const char *what)
{
  if (count < 2) {
    return 0;
  }
  uint64_t first = count;
  uint64_t second = count - 1;
  if ((first & 1) == 0) {
    first /= 2;
  } else {
    second /= 2;
  }
  return checked_multiply_u64 (first, second, what);
}

void checked_accumulate (
  uint64_t value, uint64_t &total, const char *what)
{
  if (! checked_add_u64 (total, value, total)) {
    throw AntennaM1OracleDecline (
      std::string (what) + " overflows uint64");
  }
}

void set_reason (std::string *reason, const char *message)
{
  if (! reason) {
    return;
  }
  try {
    *reason = message ? message : "unknown exception";
  } catch (...) {
    //  Diagnostics cannot turn a fail-closed decline into an exception.
  }
}

std::string digest_hex (const std::array<uint8_t, 32> &digest)
{
  std::ostringstream stream;
  stream << std::hex << std::setfill ('0');
  for (size_t i = 0; i < digest.size (); ++i) {
    stream << std::setw (2) << unsigned (digest [i]);
  }
  return stream.str ();
}

db::Coord narrow_coord (__int128 value, const char *what)
{
  if (value < __int128 (std::numeric_limits<db::Coord>::min ()) ||
      value > __int128 (std::numeric_limits<db::Coord>::max ())) {
    throw AntennaM1OracleDecline (
      std::string (what) + " escapes the host coordinate range");
  }
  return db::Coord (value);
}

db::Point transform_point (
  const CudaM1WidthSpaceContext &context, int64_t x, int64_t y)
{
  static const int matrix [8][4] = {
    { 1, 0, 0, 1 }, { 0, -1, 1, 0 },
    { -1, 0, 0, -1 }, { 0, 1, -1, 0 },
    { 1, 0, 0, -1 }, { 0, 1, 1, 0 },
    { -1, 0, 0, 1 }, { 0, -1, -1, 0 }
  };
  if (context.transform_code >= 8) {
    throw AntennaM1OracleDecline (
      "oracle context has an invalid orthogonal transform");
  }
  const int *m = matrix [context.transform_code];
  const __int128 world_x =
    __int128 (m [0]) * x + __int128 (m [1]) * y + context.tx;
  const __int128 world_y =
    __int128 (m [2]) * x + __int128 (m [3]) * y + context.ty;
  return db::Point (
    narrow_coord (world_x, "oracle polygon x"),
    narrow_coord (world_y, "oracle polygon y"));
}

db::Polygon expand_polygon (
  const CudaRawManhattanScene &scene,
  const CudaM1WidthSpaceContext &context,
  const CudaM1WidthSpacePolygon &record)
{
  std::vector<db::Point> points;
  points.reserve (record.edge_count);
  const uint64_t end = record.edge_begin + uint64_t (record.edge_count);
  if (end > scene.edges.size ()) {
    throw AntennaM1OracleDecline (
      "oracle polygon edge range escapes the capture");
  }
  for (uint64_t edge_id = record.edge_begin;
       edge_id < end; ++edge_id) {
    const CudaM1WidthSpaceEdge &edge = scene.edges [size_t (edge_id)];
    points.push_back (transform_point (context, edge.x1, edge.y1));
  }
  db::Polygon polygon;
  polygon.assign_hull (points.begin (), points.end ());
  if (polygon.vertices () == 0 || polygon.holes () != 0) {
    throw AntennaM1OracleDecline (
      "oracle polygon reconstruction changed qualified topology");
  }
  return polygon;
}

void expand_domain (
  const CudaAntennaM1Capture &capture,
  CudaAntennaM1Domain domain,
  std::vector<OraclePolygon> &output)
{
  const CudaRawManhattanScene &scene = capture.domains [domain];
  if (scene.flat_polygon_count > output.max_size ()) {
    throw AntennaM1OracleDecline (
      "oracle domain exceeds its host vector capacity");
  }
  output.reserve (size_t (scene.flat_polygon_count));
  for (size_t context_id = 0;
       context_id < scene.contexts.size (); ++context_id) {
    const CudaM1WidthSpaceContext &context =
      scene.contexts [context_id];
    const CudaM1WidthSpaceCell &cell =
      scene.cells [context.cell_id];
    for (uint32_t local = 0; local < cell.polygon_count; ++local) {
      const CudaM1WidthSpacePolygon &record =
        scene.polygons [size_t (cell.polygon_begin + local)];
      OraclePolygon occurrence;
      occurrence.polygon = expand_polygon (scene, context, record);
      occurrence.domain = uint32_t (domain);
      occurrence.context_id = uint32_t (context_id);
      occurrence.context_parent_id =
        capture.context_parent_ids [context_id];
      occurrence.dense_cell_id = context.cell_id;
      occurrence.polygon_id = record.polygon_id;
      occurrence.source_cell_index = cell.source_cell_index;
      occurrence.node_id = std::numeric_limits<uint64_t>::max ();
      output.push_back (occurrence);
    }
  }
  if (output.size () != scene.flat_polygon_count) {
    throw AntennaM1OracleDecline (
      "oracle domain expansion disagrees with the capture census");
  }
}

void count_pair (
  CudaAntennaM1Oracle &oracle, size_t relation,
  const CudaAntennaM1OracleLimits &limits)
{
  checked_accumulate (
    1, oracle.graph_candidate_pair_count,
    "oracle graph candidate-pair count");
  checked_accumulate (
    1, oracle.relation_candidate_pair_counts [relation],
    "oracle relation candidate-pair count");
  if (oracle.graph_candidate_pair_count >
      limits.max_candidate_pairs) {
    throw AntennaM1OracleDecline (
      "oracle graph candidate-pair work exceeds the configured capacity");
  }
}

void count_annotation_pair (
  CudaAntennaM1Oracle &oracle,
  const CudaAntennaM1OracleLimits &limits)
{
  checked_accumulate (
    1, oracle.annotation_candidate_pair_count,
    "oracle annotation candidate-pair count");
  uint64_t total = 0;
  if (! checked_add_u64 (
        oracle.graph_candidate_pair_count,
        oracle.annotation_candidate_pair_count, total) ||
      total > limits.max_candidate_pairs) {
    throw AntennaM1OracleDecline (
      "oracle total candidate-pair work exceeds the configured capacity");
  }
}

void scan_self_relation (
  const std::vector<OraclePolygon> &nodes, size_t relation,
  const CudaAntennaM1OracleLimits &limits,
  DeterministicDisjointSet &sets,
  CudaAntennaM1Oracle &oracle)
{
  for (size_t first = 0; first < nodes.size (); ++first) {
    for (size_t second = first + 1;
         second < nodes.size (); ++second) {
      count_pair (oracle, relation, limits);
      if (db::interact (
            nodes [first].polygon, nodes [second].polygon)) {
        checked_accumulate (
          1, oracle.graph_edge_count, "oracle graph edge count");
        checked_accumulate (
          1, oracle.relation_edge_counts [relation],
          "oracle relation edge count");
        sets.unite (
          nodes [first].node_id, nodes [second].node_id);
      }
    }
  }
}

void scan_cross_relation (
  const std::vector<OraclePolygon> &first,
  const std::vector<OraclePolygon> &second, size_t relation,
  const CudaAntennaM1OracleLimits &limits,
  DeterministicDisjointSet &sets,
  CudaAntennaM1Oracle &oracle)
{
  for (size_t a = 0; a < first.size (); ++a) {
    for (size_t b = 0; b < second.size (); ++b) {
      count_pair (oracle, relation, limits);
      if (db::interact (first [a].polygon, second [b].polygon)) {
        checked_accumulate (
          1, oracle.graph_edge_count, "oracle graph edge count");
        checked_accumulate (
          1, oracle.relation_edge_counts [relation],
          "oracle relation edge count");
        sets.unite (first [a].node_id, second [b].node_id);
      }
    }
  }
}

db::Region domain_region (
  const std::vector<OraclePolygon> &polygons)
{
  db::Region region;
  region.set_merged_semantics (true);
  for (std::vector<OraclePolygon>::const_iterator polygon =
         polygons.begin (); polygon != polygons.end (); ++polygon) {
    region.insert (polygon->polygon);
  }
  return region;
}

std::vector<db::Polygon> merged_polygons (const db::Region &region)
{
  std::vector<db::Polygon> polygons;
  for (db::Region::const_iterator polygon = region.begin_merged ();
       ! polygon.at_end (); ++polygon) {
    polygons.push_back (*polygon);
  }
  return polygons;
}

uint64_t positive_area (const db::Polygon &polygon, const char *what)
{
  const __int128 area = __int128 (polygon.area ());
  if (area <= 0) {
    throw AntennaM1OracleDecline (
      std::string (what) + " produced a non-positive polygon");
  }
  if (area >
      __int128 (std::numeric_limits<uint64_t>::max ())) {
    throw AntennaM1OracleDecline (
      std::string (what) + " area exceeds uint64");
  }
  return uint64_t (area);
}

void assign_component_area (
  const std::vector<OraclePolygon> &nodes,
  const std::vector<db::Polygon> &annotation,
  const CudaAntennaM1OracleLimits &limits,
  CudaAntennaM1Oracle &oracle,
  std::vector<uint64_t> &component_area,
  uint64_t &total_area,
  const char *what)
{
  for (size_t shape = 0; shape < annotation.size (); ++shape) {
    uint64_t assigned =
      std::numeric_limits<uint64_t>::max ();
    for (size_t node = 0; node < nodes.size (); ++node) {
      count_annotation_pair (oracle, limits);
      if (! db::interact (
            nodes [node].polygon, annotation [shape])) {
        continue;
      }
      const uint64_t root =
        oracle.canonical_labels [size_t (nodes [node].node_id)];
      if (assigned == std::numeric_limits<uint64_t>::max ()) {
        assigned = root;
      } else if (assigned != root) {
        throw AntennaM1OracleDecline (
          std::string (what) +
          " annotation touches more than one graph component");
      }
    }
    if (assigned == std::numeric_limits<uint64_t>::max ()) {
      throw AntennaM1OracleDecline (
        std::string (what) +
        " annotation cannot be assigned to a graph component");
    }
    const uint64_t area = positive_area (annotation [shape], what);
    checked_accumulate (
      area, component_area [size_t (assigned)],
      "oracle component annotation area");
    checked_accumulate (area, total_area, what);
  }
}

void assign_diode_exemptions (
  const std::vector<OraclePolygon> &contacts,
  const std::vector<db::Polygon> &diodes,
  const CudaAntennaM1OracleLimits &limits,
  CudaAntennaM1Oracle &oracle,
  std::vector<uint8_t> &exempt)
{
  for (size_t contact = 0; contact < contacts.size (); ++contact) {
    for (size_t diode = 0; diode < diodes.size (); ++diode) {
      count_annotation_pair (oracle, limits);
      if (db::interact (
            contacts [contact].polygon, diodes [diode])) {
        const uint64_t root =
          oracle.canonical_labels [
            size_t (contacts [contact].node_id)];
        exempt [size_t (root)] = 1;
      }
    }
  }
}

std::array<uint8_t, 32> digest_node_identities (
  const CudaAntennaM1Capture &capture,
  const std::vector<const OraclePolygon *> &nodes)
{
  static const char magic [8] =
    { 'K', 'A', 'N', 'T', 'O', 'I', 'D', '1' };
  CanonicalDigest sha;
  sha.bytes (magic, sizeof (magic));
  sha.u32 (oracle_format_version);
  sha.bytes (
    capture.hierarchy_digest.data (), capture.hierarchy_digest.size ());
  sha.u32 (3);
  for (size_t selected = 0; selected < 3; ++selected) {
    const CudaAntennaM1Domain domain =
      graph_domain_order [selected];
    sha.u32 (uint32_t (domain));
    sha.bytes (
      capture.domains [domain].digest.data (),
      capture.domains [domain].digest.size ());
  }
  sha.u64 (nodes.size ());
  for (std::vector<const OraclePolygon *>::const_iterator node =
         nodes.begin (); node != nodes.end (); ++node) {
    const OraclePolygon &value = **node;
    const CudaM1WidthSpaceContext &context =
      capture.domains [value.domain].contexts [value.context_id];
    sha.u32 (value.domain);
    sha.u32 (value.context_id);
    sha.u32 (value.context_parent_id);
    sha.u32 (value.dense_cell_id);
    sha.u64 (value.source_cell_index);
    sha.u32 (value.polygon_id);
    sha.i64 (context.tx);
    sha.i64 (context.ty);
    sha.u32 (context.transform_code);
  }
  return sha.finish ();
}

std::array<uint8_t, 32> digest_partition (
  const CudaAntennaM1Oracle &oracle)
{
  static const char magic [8] =
    { 'K', 'A', 'N', 'T', 'P', '1', '0', '1' };
  CanonicalDigest sha;
  sha.bytes (magic, sizeof (magic));
  sha.u32 (oracle.format_version);
  sha.bytes (
    oracle.node_identity_digest.data (),
    oracle.node_identity_digest.size ());
  sha.u64 (oracle.graph_node_count);
  sha.u64 (oracle.component_count);
  sha.u64 (oracle.canonical_root_count);
  sha.u64 (oracle.canonical_labels.size ());
  for (std::vector<uint64_t>::const_iterator label =
         oracle.canonical_labels.begin ();
       label != oracle.canonical_labels.end (); ++label) {
    sha.u64 (*label);
  }
  return sha.finish ();
}

std::array<uint8_t, 32> digest_connectivity (
  const CudaAntennaM1Oracle &oracle)
{
  static const char magic [8] =
    { 'K', 'A', 'N', 'T', 'G', '1', '0', '1' };
  CanonicalDigest sha;
  sha.bytes (magic, sizeof (magic));
  sha.u32 (oracle.format_version);
  sha.bytes (
    oracle.node_identity_digest.data (),
    oracle.node_identity_digest.size ());
  sha.u64 (oracle.graph_node_count);
  sha.u64 (oracle.component_count);
  sha.u64 (oracle.canonical_root_count);
  for (size_t relation = 0;
       relation < CudaAntennaM1OracleRelationCount; ++relation) {
    sha.u32 (uint32_t (relation));
    sha.u64 (oracle.relation_candidate_pair_counts [relation]);
    sha.u64 (oracle.relation_edge_counts [relation]);
  }
  sha.u64 (oracle.canonical_labels.size ());
  for (std::vector<uint64_t>::const_iterator label =
         oracle.canonical_labels.begin ();
       label != oracle.canonical_labels.end (); ++label) {
    sha.u64 (*label);
  }
  return sha.finish ();
}

std::array<uint8_t, 32> digest_annotations (
  const CudaAntennaM1Capture &capture,
  const CudaAntennaM1Oracle &oracle,
  const std::vector<uint64_t> &gate_area,
  const std::vector<uint64_t> &metal1_area,
  const std::vector<uint8_t> &diode_exempt)
{
  static const char magic [8] =
    { 'K', 'A', 'N', 'T', 'A', '1', '0', '1' };
  CanonicalDigest sha;
  sha.bytes (magic, sizeof (magic));
  sha.u32 (oracle.format_version);
  sha.bytes (
    oracle.partition_digest.data (),
    oracle.partition_digest.size ());
  static const CudaAntennaM1Domain annotation_domains [3] = {
    CudaAntennaM1Active,
    CudaAntennaM1Nplus,
    CudaAntennaM1Nwell
  };
  sha.u32 (3);
  for (size_t selected = 0; selected < 3; ++selected) {
    const CudaAntennaM1Domain domain =
      annotation_domains [selected];
    sha.u32 (uint32_t (domain));
    sha.bytes (
      capture.domains [domain].digest.data (),
      capture.domains [domain].digest.size ());
  }
  sha.u64 (oracle.component_count);
  for (size_t node = 0;
       node < oracle.canonical_labels.size (); ++node) {
    if (oracle.canonical_labels [node] != node) {
      continue;
    }
    sha.u64 (node);
    sha.u64 (gate_area [node]);
    sha.u64 (metal1_area [node]);
    sha.u32 (diode_exempt [node] ? 1 : 0);
  }
  return sha.finish ();
}

std::array<uint8_t, 32> digest_oracle (
  const CudaAntennaM1Capture &capture,
  const CudaAntennaM1Oracle &oracle)
{
  static const char magic [8] =
    { 'K', 'A', 'N', 'T', 'O', '1', '0', '1' };
  CanonicalDigest sha;
  sha.bytes (magic, sizeof (magic));
  sha.u32 (oracle.format_version);
  sha.bytes (capture.digest.data (), capture.digest.size ());
  sha.bytes (
    oracle.node_identity_digest.data (),
    oracle.node_identity_digest.size ());
  sha.bytes (
    oracle.partition_digest.data (),
    oracle.partition_digest.size ());
  sha.bytes (
    oracle.connectivity_digest.data (),
    oracle.connectivity_digest.size ());
  sha.bytes (
    oracle.annotation_digest.data (),
    oracle.annotation_digest.size ());
  sha.u64 (oracle.context_count);
  sha.u64 (oracle.graph_node_count);
  sha.u64 (oracle.component_count);
  sha.u64 (oracle.canonical_root_count);
  sha.u64 (oracle.top_context_component_count);
  sha.u64 (oracle.graph_candidate_pair_count);
  sha.u64 (oracle.graph_edge_count);
  sha.u64 (oracle.annotation_candidate_pair_count);
  sha.u64 (oracle.gate_component_count);
  sha.u64 (oracle.diode_exempt_component_count);
  sha.u64 (oracle.gate_area_dbu2);
  sha.u64 (oracle.metal1_area_dbu2);
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    sha.u64 (oracle.domain_node_counts [domain]);
    sha.u64 (oracle.domain_component_counts [domain]);
  }
  return sha.finish ();
}

} // anonymous namespace

CudaAntennaM1OracleLimits::CudaAntennaM1OracleLimits ()
  : max_total_flat_polygons (default_max_total_flat_polygons),
    max_total_flat_edges (default_max_total_flat_edges),
    max_graph_nodes (default_max_graph_nodes),
    max_candidate_pairs (default_max_candidate_pairs)
{
  //  nothing yet
}

CudaAntennaM1Oracle::CudaAntennaM1Oracle ()
  : format_version (oracle_format_version),
    reserved (0),
    context_count (0),
    graph_node_count (0),
    component_count (0),
    canonical_root_count (0),
    top_context_component_count (0),
    graph_candidate_pair_count (0),
    graph_edge_count (0),
    annotation_candidate_pair_count (0),
    gate_component_count (0),
    diode_exempt_component_count (0),
    gate_area_dbu2 (0),
    metal1_area_dbu2 (0),
    domain_node_counts (),
    domain_component_counts (),
    relation_candidate_pair_counts (),
    relation_edge_counts (),
    canonical_labels (),
    node_identity_digest (),
    partition_digest (),
    connectivity_digest (),
    annotation_digest (),
    oracle_digest ()
{
  domain_node_counts.fill (0);
  domain_component_counts.fill (0);
  relation_candidate_pair_counts.fill (0);
  relation_edge_counts.fill (0);
  node_identity_digest.fill (0);
  partition_digest.fill (0);
  connectivity_digest.fill (0);
  annotation_digest.fill (0);
  oracle_digest.fill (0);
}

bool cuda_antenna_m1_cpu_oracle (
  const CudaAntennaM1Capture &capture,
  const CudaAntennaM1OracleLimits &limits,
  CudaAntennaM1Oracle &oracle,
  std::string *decline_reason)
{
  try {
    if (! limits.max_total_flat_polygons ||
        ! limits.max_total_flat_edges ||
        ! limits.max_graph_nodes ||
        ! limits.max_candidate_pairs) {
      throw AntennaM1OracleDecline (
        "an antenna oracle capacity is zero");
    }

    CudaAntennaM1Census census;
    std::string census_reason;
    if (! cuda_antenna_m1_capture_census (
          capture, census, &census_reason)) {
      throw AntennaM1OracleDecline (
        std::string ("antenna oracle capture is invalid: ") +
        census_reason);
    }

    uint64_t total_flat_polygons = 0;
    uint64_t graph_nodes = 0;
    for (size_t domain = 0;
         domain < CudaAntennaM1DomainCount; ++domain) {
      checked_accumulate (
        capture.domains [domain].flat_polygon_count,
        total_flat_polygons, "oracle total flat polygon count");
    }
    for (size_t selected = 0; selected < 3; ++selected) {
      checked_accumulate (
        capture.domains [graph_domain_order [selected]].
          flat_polygon_count,
        graph_nodes, "oracle graph node count");
    }
    if (total_flat_polygons > limits.max_total_flat_polygons) {
      throw AntennaM1OracleDecline (
        "oracle total flat polygons exceed the configured capacity");
    }
    if (census.expanded_edge_count > limits.max_total_flat_edges) {
      throw AntennaM1OracleDecline (
        "oracle total expanded edges exceed the configured capacity");
    }
    if (graph_nodes > limits.max_graph_nodes ||
        graph_nodes > std::numeric_limits<size_t>::max ()) {
      throw AntennaM1OracleDecline (
        "oracle graph nodes exceed the configured capacity");
    }

    const uint64_t poly_nodes =
      capture.domains [CudaAntennaM1Poly].flat_polygon_count;
    const uint64_t contact_nodes =
      capture.domains [CudaAntennaM1Contact].flat_polygon_count;
    const uint64_t metal1_nodes =
      capture.domains [CudaAntennaM1Metal1].flat_polygon_count;
    uint64_t expected_graph_pairs = 0;
    checked_accumulate (
      checked_choose_two (poly_nodes, "oracle POLY self-pair work"),
      expected_graph_pairs, "oracle graph pair work");
    checked_accumulate (
      checked_choose_two (
        contact_nodes, "oracle CONTACT self-pair work"),
      expected_graph_pairs, "oracle graph pair work");
    checked_accumulate (
      checked_choose_two (
        metal1_nodes, "oracle M1 self-pair work"),
      expected_graph_pairs, "oracle graph pair work");
    checked_accumulate (
      checked_multiply_u64 (
        poly_nodes, contact_nodes, "oracle POLY-CONTACT pair work"),
      expected_graph_pairs, "oracle graph pair work");
    checked_accumulate (
      checked_multiply_u64 (
        contact_nodes, metal1_nodes, "oracle CONTACT-M1 pair work"),
      expected_graph_pairs, "oracle graph pair work");
    if (expected_graph_pairs > limits.max_candidate_pairs) {
      throw AntennaM1OracleDecline (
        "oracle graph candidate-pair work exceeds the configured capacity");
    }

    CudaAntennaM1Oracle candidate;
    candidate.context_count = census.shared_context_count;
    candidate.graph_node_count = graph_nodes;

    std::array<
      std::vector<OraclePolygon>, CudaAntennaM1DomainCount> domains;
    for (size_t domain = 0;
         domain < CudaAntennaM1DomainCount; ++domain) {
      expand_domain (
        capture, CudaAntennaM1Domain (domain), domains [domain]);
      candidate.domain_node_counts [domain] =
        domains [domain].size ();
    }

    std::vector<const OraclePolygon *> nodes;
    nodes.reserve (size_t (graph_nodes));
    uint64_t next_node = 0;
    for (size_t selected = 0; selected < 3; ++selected) {
      std::vector<OraclePolygon> &domain =
        domains [graph_domain_order [selected]];
      for (std::vector<OraclePolygon>::iterator polygon =
             domain.begin (); polygon != domain.end (); ++polygon) {
        polygon->node_id = next_node++;
        nodes.push_back (&*polygon);
      }
    }
    if (next_node != graph_nodes || nodes.size () != graph_nodes) {
      throw AntennaM1OracleDecline (
        "oracle graph-node enumeration is inconsistent");
    }

    candidate.node_identity_digest =
      digest_node_identities (capture, nodes);

    DeterministicDisjointSet sets { size_t (graph_nodes) };
    scan_self_relation (
      domains [CudaAntennaM1Poly],
      CudaAntennaM1OraclePolySelf,
      limits, sets, candidate);
    scan_self_relation (
      domains [CudaAntennaM1Contact],
      CudaAntennaM1OracleContactSelf,
      limits, sets, candidate);
    scan_self_relation (
      domains [CudaAntennaM1Metal1],
      CudaAntennaM1OracleMetal1Self,
      limits, sets, candidate);
    scan_cross_relation (
      domains [CudaAntennaM1Poly],
      domains [CudaAntennaM1Contact],
      CudaAntennaM1OraclePolyContact,
      limits, sets, candidate);
    scan_cross_relation (
      domains [CudaAntennaM1Contact],
      domains [CudaAntennaM1Metal1],
      CudaAntennaM1OracleContactMetal1,
      limits, sets, candidate);
    if (candidate.graph_candidate_pair_count !=
        expected_graph_pairs) {
      throw AntennaM1OracleDecline (
        "oracle graph pair enumeration disagrees with its census");
    }

    candidate.canonical_labels.resize (size_t (graph_nodes));
    std::set<uint64_t> roots;
    std::set<uint64_t> top_context_roots;
    std::array<std::set<uint64_t>, CudaAntennaM1DomainCount>
      domain_roots;
    for (size_t node = 0; node < nodes.size (); ++node) {
      const uint64_t root = sets.find (uint64_t (node));
      candidate.canonical_labels [node] = root;
      roots.insert (root);
      domain_roots [nodes [node]->domain].insert (root);
      if (nodes [node]->context_id == 0) {
        top_context_roots.insert (root);
      }
    }
    candidate.component_count = roots.size ();
    candidate.canonical_root_count = 0;
    for (size_t node = 0; node < nodes.size (); ++node) {
      if (candidate.canonical_labels [node] == node) {
        ++candidate.canonical_root_count;
      }
    }
    if (candidate.component_count !=
        candidate.canonical_root_count) {
      throw AntennaM1OracleDecline (
        "oracle canonical roots disagree with component count");
    }
    candidate.top_context_component_count =
      top_context_roots.size ();
    for (size_t domain = 0;
         domain < CudaAntennaM1DomainCount; ++domain) {
      candidate.domain_component_counts [domain] =
        domain_roots [domain].size ();
    }
    candidate.partition_digest =
      digest_partition (candidate);
    candidate.connectivity_digest =
      digest_connectivity (candidate);

    const db::Region poly =
      domain_region (domains [CudaAntennaM1Poly]);
    const db::Region active =
      domain_region (domains [CudaAntennaM1Active]);
    const db::Region nplus =
      domain_region (domains [CudaAntennaM1Nplus]);
    const db::Region nwell =
      domain_region (domains [CudaAntennaM1Nwell]);
    const db::Region metal1 =
      domain_region (domains [CudaAntennaM1Metal1]);

    const std::vector<db::Polygon> gate =
      merged_polygons (poly & active);
    const std::vector<db::Polygon> merged_metal1 =
      merged_polygons (metal1);
    const std::vector<db::Polygon> diode =
      merged_polygons (nplus & (active - nwell));

    std::vector<uint64_t> gate_area (size_t (graph_nodes), 0);
    std::vector<uint64_t> metal1_area (size_t (graph_nodes), 0);
    std::vector<uint8_t> diode_exempt (size_t (graph_nodes), 0);
    assign_component_area (
      domains [CudaAntennaM1Poly], gate, limits, candidate,
      gate_area, candidate.gate_area_dbu2, "gate");
    assign_component_area (
      domains [CudaAntennaM1Metal1], merged_metal1, limits,
      candidate, metal1_area, candidate.metal1_area_dbu2, "M1");
    assign_diode_exemptions (
      domains [CudaAntennaM1Contact], diode, limits,
      candidate, diode_exempt);

    for (std::set<uint64_t>::const_iterator root = roots.begin ();
         root != roots.end (); ++root) {
      if (gate_area [size_t (*root)] > 1) {
        ++candidate.gate_component_count;
      }
      if (diode_exempt [size_t (*root)]) {
        ++candidate.diode_exempt_component_count;
      }
    }
    candidate.annotation_digest = digest_annotations (
      capture, candidate, gate_area, metal1_area, diode_exempt);
    candidate.oracle_digest = digest_oracle (capture, candidate);

    oracle = std::move (candidate);
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (decline_reason, "unknown exception");
  }
  return false;
}

std::string cuda_antenna_m1_oracle_text (
  const CudaAntennaM1Oracle &oracle)
{
  static const char *domain_labels [CudaAntennaM1DomainCount] = {
    "POLY", "ACTIVE", "NPLUS", "NWELL", "CONTACT", "M1"
  };
  static const char *relation_labels [
    CudaAntennaM1OracleRelationCount] = {
    "POLY-POLY", "CONTACT-CONTACT", "M1-M1",
    "POLY-CONTACT", "CONTACT-M1"
  };
  std::ostringstream stream;
  stream
    << "antenna_m1_cpu_conductor_oracle"
    << " format=" << oracle.format_version
    << " contexts=" << oracle.context_count
    << " nodes=" << oracle.graph_node_count
    << " components=" << oracle.component_count
    << " canonical_roots=" << oracle.canonical_root_count
    << " top_context_components="
    << oracle.top_context_component_count
    << " graph_pairs=" << oracle.graph_candidate_pair_count
    << " graph_edges=" << oracle.graph_edge_count
    << " annotation_pairs="
    << oracle.annotation_candidate_pair_count
    << " gate_components=" << oracle.gate_component_count
    << " diode_exempt_components="
    << oracle.diode_exempt_component_count
    << " gate_area_dbu2=" << oracle.gate_area_dbu2
    << " metal1_area_dbu2=" << oracle.metal1_area_dbu2;
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    stream
      << " " << domain_labels [domain]
      << "{nodes=" << oracle.domain_node_counts [domain];
    if (domain == CudaAntennaM1Poly ||
        domain == CudaAntennaM1Contact ||
        domain == CudaAntennaM1Metal1) {
      stream
        << ",components="
        << oracle.domain_component_counts [domain];
    } else {
      stream << ",components=n/a";
    }
    stream << "}";
  }
  for (size_t relation = 0;
       relation < CudaAntennaM1OracleRelationCount; ++relation) {
    stream
      << " " << relation_labels [relation]
      << "{pairs=" << oracle.relation_candidate_pair_counts [relation]
      << ",edges=" << oracle.relation_edge_counts [relation] << "}";
  }
  stream
    << " identity_sha256="
    << digest_hex (oracle.node_identity_digest)
    << " partition_sha256="
    << digest_hex (oracle.partition_digest)
    << " connectivity_sha256="
    << digest_hex (oracle.connectivity_digest)
    << " annotation_sha256="
    << digest_hex (oracle.annotation_digest)
    << " oracle_sha256=" << digest_hex (oracle.oracle_digest);
  return stream.str ();
}

} // namespace db
