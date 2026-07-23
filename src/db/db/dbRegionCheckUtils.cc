
/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

*/


#include "dbRegionCheckUtils.h"
#include "dbPolygonTools.h"
#include "dbEdgeBoolean.h"
#include "tlSelect.h"
#include "tlEnv.h"
#include "tlFileUtils.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>
#include <unordered_map>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#endif

namespace db
{

// -------------------------------------------------------------------------------------
//  Edge2EdgeCheckBase implementation

Edge2EdgeCheckBase::Edge2EdgeCheckBase (const EdgeRelationFilter &check, bool different_polygons, bool requires_different_layers, bool with_shielding, bool symmetric_edges)
  : mp_check (&check), m_requires_different_layers (requires_different_layers), m_different_polygons (different_polygons),
    m_first_pseudo (std::numeric_limits<size_t>::max ()),
    m_with_shielding (with_shielding),
    m_symmetric_edges (symmetric_edges),
    m_has_edge_pair_output (true),
    m_has_negative_edge_output (false),
    m_pass (0)
{
  m_distance = check.distance ();
}

bool
Edge2EdgeCheckBase::prepare_next_pass ()
{
  ++m_pass;

  if (m_pass == 1) {

    m_first_pseudo = m_ep.size ();

    if (m_with_shielding && ! m_ep.empty ()) {

      m_ep_discarded.resize (m_ep.size (), false);

      //  second pass:
      return true;

    } else if (m_has_negative_edge_output) {

      //  second pass:
      return true;

    }

  }

  if (! m_ep.empty () && m_has_edge_pair_output) {

    std::vector<bool>::const_iterator d = m_ep_discarded.begin ();
    std::vector<bool>::const_iterator i = m_ep_intra_polygon.begin ();
    std::vector<db::EdgePair>::const_iterator ep = m_ep.begin ();
    while (ep != m_ep.end () && size_t (ep - m_ep.begin ()) < m_first_pseudo) {
      bool use_result = true;
      if (d != m_ep_discarded.end ()) {
        use_result = ! *d;
        ++d;
      }
      if (use_result) {
        put (*ep, *i);
      }
      ++ep;
      ++i;
    }

  }

  return false;
}

static inline bool shields (const db::EdgePair &ep, const db::Edge &q)
{
  db::Edge pe1 (ep.first ().p1 (), ep.second ().p2 ());
  db::Edge pe2 (ep.second ().p1 (), ep.first ().p2 ());

  std::pair<bool, db::Point> ip1 = pe1.intersect_point (q);
  std::pair<bool, db::Point> ip2 = pe2.intersect_point (q);

  if (ip1.first && ip2.first) {
    return ip1.second != ip2.second || (pe1.side_of (q.p1 ()) != 0 && pe2.side_of (q.p2 ()) != 0);
  } else {
    return false;
  }
}

void
Edge2EdgeCheckBase::finish (const Edge *o, size_t p)
{
  if (m_has_negative_edge_output && m_pass == 1 && m_pseudo_edges.find (std::make_pair (*o, p)) == m_pseudo_edges.end ()) {

    std::pair<db::Edge, size_t> k (*o, p);
    std::multimap<std::pair<db::Edge, size_t>, size_t>::const_iterator i0 = m_e2ep.find (k);

    bool fully_removed = false;
    bool any = false;
    for (std::multimap<std::pair<db::Edge, size_t>, size_t>::const_iterator i = i0; ! fully_removed && i != m_e2ep.end () && i->first == k; ++i) {
      size_t n = i->second / 2;
      if (n >= m_ep_discarded.size () || !m_ep_discarded [n]) {
        any = true;
        fully_removed = (((i->second & 1) == 0 ? m_ep [n].first () : m_ep [n].second ()) == *o);
      }
    }

    if (! any) {

      put_negative (*o, (int) p);

    } else if (! fully_removed) {

      std::set<db::Edge> partial_edges;

      db::EdgeBooleanCluster<std::set<db::Edge> > ec (&partial_edges, 0, db::EdgeNot);
      ec.add (o, 0);

      for (std::multimap<std::pair<db::Edge, size_t>, size_t>::const_iterator i = i0; i != m_e2ep.end () && i->first == k; ++i) {
        size_t n = i->second / 2;
        if (n >= m_ep_discarded.size () || !m_ep_discarded [n]) {
          ec.add (((i->second & 1) == 0 ? &m_ep [n].first () : &m_ep [n].second ()), 1);
        }
      }

      ec.finish ();

      for (std::set<db::Edge>::const_iterator e = partial_edges.begin (); e != partial_edges.end (); ++e) {
        put_negative (*e, (int) p);
      }

    }

  }
}

bool
Edge2EdgeCheckBase::feed_pseudo_edges (db::box_scanner<db::Edge, size_t> &scanner)
{
  if (m_pass == 1) {
    for (std::set<std::pair<db::Edge, size_t> >::const_iterator e = m_pseudo_edges.begin (); e != m_pseudo_edges.end (); ++e) {
      scanner.insert (&e->first, e->second);
    }
    return ! m_pseudo_edges.empty ();
  } else {
    return false;
  }
}

inline bool edges_considered (bool requires_different_polygons, bool requires_different_layers, size_t p1, size_t p2)
{
  if (p1 == p2) {
    if (requires_different_polygons) {
      return false;
    } else if ((p1 & size_t (1)) != 0) {
      //  edges from the same polygon are only considered on first layer.
      //  Reasoning: this case happens when "intruder" polygons are put on layer 1
      //  while "subject" polygons are put on layer 0. We don't want "intruders"
      //  to generate intra-polygon markers.
      return false;
    }
  }

  if (((p1 ^ p2) & size_t (1)) == 0) {
    if (requires_different_layers) {
      return false;
    } else if ((p1 & size_t (1)) != 0) {
      //  edges on the same layer are only considered on first layer.
      //  Reasoning: this case happens when "intruder" polygons are put on layer 1
      //  while "subject" polygons are put on layer 0. We don't want "intruders"
      //  to generate inter-polygon markers between them.
      return false;
    }
  }

  return true;
}

void
Edge2EdgeCheckBase::add (const db::Edge *o1, size_t p1, const db::Edge *o2, size_t p2)
{
  if (m_pass == 0) {

    //  Overlap or inside checks require input from different layers
    if (edges_considered (m_different_polygons, m_requires_different_layers, p1, p2)) {

      //  ensure that the first check argument is of layer 1 and the second of
      //  layer 2 (unless both are of the same layer)
      int l1 = int (p1 & size_t (1));
      int l2 = int (p2 & size_t (1));

      if (l1 > l2) {
        std::swap (o1, o2);
        std::swap (p1, p2);
      }

      db::EdgePair ep;
      if (mp_check->check (*o1, *o2, &ep)) {

        ep.set_symmetric (m_symmetric_edges);

        //  found a violation: store inside the local buffer for now. In the second
        //  pass we will eliminate those which are shielded completely (with shielding)
        //  and/or compute the negative edges.
        size_t n = m_ep.size ();

        m_ep.push_back (ep);
        m_ep_intra_polygon.push_back (p1 == p2);

        m_e2ep.insert (std::make_pair (std::make_pair (*o1, p1), n * 2));
        m_e2ep.insert (std::make_pair (std::make_pair (*o2, p2), n * 2 + 1));

        if (m_has_negative_edge_output) {

          bool antiparallel = (mp_check->relation () == WidthRelation || mp_check->relation () == SpaceRelation);

          //  pseudo1 and pseudo2 are the connecting edges of the edge pairs. Together with the
          //  original edges they form a quadrangle.
          db::Edge pseudo1 (ep.first ().p1 (), antiparallel ? ep.second ().p2 () : ep.second ().p1 ());
          db::Edge pseudo2 (antiparallel ? ep.second ().p1 () : ep.second ().p2 (), ep.first ().p2 ());

          m_pseudo_edges.insert (std::make_pair (pseudo1, p1));
          m_pseudo_edges.insert (std::make_pair (pseudo2, p1));
          if (p1 != p2) {
            m_pseudo_edges.insert (std::make_pair (pseudo1, p2));
            m_pseudo_edges.insert (std::make_pair (pseudo2, p2));
          }

        }

      }

    }

  } else {

    //  set the discarded flags for shielded output
    if (m_with_shielding) {

      //  a simple (complete) shielding implementation which is based on the
      //  assumption that shielding is relevant as soon as a foreign edge cuts through
      //  both of the edge pair's connecting edges.

      //  TODO: this implementation does not take into account the nature of the
      //  EdgePair - because of "whole_edge" it may not reflect the part actually
      //  violating the distance.

      std::vector<size_t> n1, n2;

      for (unsigned int p = 0; p < 2; ++p) {

        std::pair<db::Edge, size_t> k (*o1, p1);
        for (std::multimap<std::pair<db::Edge, size_t>, size_t>::const_iterator i = m_e2ep.find (k); i != m_e2ep.end () && i->first == k; ++i) {
          size_t n = i->second / 2;
          if (n < m_first_pseudo && ! m_ep_discarded [n]) {
            n1.push_back (n);
          }
        }

        std::sort (n1.begin (), n1.end ());

        std::swap (o1, o2);
        std::swap (p1, p2);
        n1.swap (n2);

      }

      for (unsigned int p = 0; p < 2; ++p) {

        std::vector<size_t> nn;
        std::set_difference (n1.begin (), n1.end (), n2.begin (), n2.end (), std::back_inserter (nn));

        for (std::vector<size_t>::const_iterator i = nn.begin (); i != nn.end (); ++i) {
          db::EdgePair ep = m_ep [*i].normalized ();
          if (shields (ep, *o2)) {
            m_ep_discarded [*i] = true;
          }
        }

        std::swap (o1, o2);
        std::swap (p1, p2);
        n1.swap (n2);

      }

    }

    //  for negative output edges are cancelled by short interactions perpendicular to them
    //  For this we have generated "pseudo edges" running along the sides of the original violation. We now check a real
    //  edge vs. a pseudo edge with the same conditions as the normal interaction and add them to the results. In the
    //  negative case this means we cancel a real edge.

    if (m_has_negative_edge_output &&
      (m_pseudo_edges.find (std::make_pair (*o1, p1)) != m_pseudo_edges.end ()) != (m_pseudo_edges.find (std::make_pair (*o2, p2)) != m_pseudo_edges.end ())) {

      //  Overlap or inside checks require input from different layers
      if (edges_considered (m_different_polygons, m_requires_different_layers, p1, p2)) {

        //  ensure that the first check argument is of layer 1 and the second of
        //  layer 2 (unless both are of the same layer)
        int l1 = int (p1 & size_t (1));
        int l2 = int (p2 & size_t (1));

        if (l1 > l2) {
          std::swap (o1, o2);
          std::swap (p1, p2);
        }

        db::EdgePair ep;
        if (mp_check->check (*o1, *o2, &ep)) {

          size_t n = m_ep.size ();

          m_ep.push_back (ep);
          m_ep_intra_polygon.push_back (p1 == p2);  //  not really required, but there for consistency

          m_e2ep.insert (std::make_pair (std::make_pair (*o1, p1), n * 2));
          m_e2ep.insert (std::make_pair (std::make_pair (*o2, p2), n * 2 + 1));

        }

      }

    }

  }

}

/**
 *  @brief Gets a value indicating whether the check requires different layers
 */
bool
Edge2EdgeCheckBase::requires_different_layers () const
{
  return m_requires_different_layers;
}

/**
 *  @brief Sets a value indicating whether the check requires different layers
 */
void
Edge2EdgeCheckBase::set_requires_different_layers (bool f)
{
  m_requires_different_layers = f;
}

/**
 *  @brief Gets a value indicating whether the check requires different layers
 */
bool
Edge2EdgeCheckBase::different_polygons () const
{
  return m_different_polygons;
}

/**
 *  @brief Sets a value indicating whether the check requires different layers
 */
void
Edge2EdgeCheckBase::set_different_polygons (bool f)
{
  m_different_polygons = f;
}

/**
 *  @brief Gets the distance value
 */
EdgeRelationFilter::distance_type
Edge2EdgeCheckBase::distance () const
{
  return m_distance;
}

bool
Edge2EdgeCheckBase::edge_replay_capture_eligible () const
{
  return m_pass == 0 &&
         m_has_edge_pair_output && ! m_has_negative_edge_output &&
         m_with_shielding &&
         m_different_polygons && m_requires_different_layers &&
         mp_check->relation () == db::OverlapRelation &&
         mp_check->metrics () == db::Projection &&
         mp_check->ignore_angle () == 90.0 &&
         mp_check->min_projection () == 0 &&
         mp_check->max_projection () == std::numeric_limits<EdgeRelationFilter::distance_type>::max () &&
         ! mp_check->whole_edges () &&
         mp_check->get_zero_distance_mode () == db::IncludeZeroDistanceWhenTouching;
}

bool
Edge2EdgeCheckBase::edge_replay_capture_exact_accepts (const db::Edge &edge1, size_t property1, const db::Edge &edge2, size_t property2) const
{
  if (! edge_replay_capture_eligible () ||
      ! edges_considered (m_different_polygons, m_requires_different_layers, property1, property2)) {
    return false;
  }

  const db::Edge *o1 = &edge1;
  const db::Edge *o2 = &edge2;
  size_t p1 = property1;
  size_t p2 = property2;

  //  Match add(): the exact relation is directional and layer zero must be
  //  presented as the first argument.
  if ((p1 & size_t (1)) > (p2 & size_t (1))) {
    std::swap (o1, o2);
    std::swap (p1, p2);
  }

  return mp_check->check (*o1, *o2, 0);
}

// -------------------------------------------------------------------------------------
//  Poly2PolyCheckBase implementation

namespace
{

enum EdgeReplayRecordFlag
{
  EdgeReplayHasEndpoints = 1u << 0,
  EdgeReplaySideB = 1u << 1
};

enum EdgeReplayCaptureFlag
{
  EdgeReplayBroadPairsSortedUnique = 1u << 0,
  EdgeReplayExactPairsSortedUnique = 1u << 1,
  EdgeReplayPropertyIsUint64 = 1u << 2,
  EdgeReplayNegativeOneIsInfiniteProjection = 1u << 3
};

enum EdgeReplayOptionFlag
{
  EdgeReplayDifferentPolygons = 1u << 0,
  EdgeReplayDifferentLayers = 1u << 1,
  EdgeReplayShielded = 1u << 2,
  EdgeReplayPositiveOutput = 1u << 3
};

struct EdgeReplayCaptureConfig
{
  EdgeReplayCaptureConfig ()
    : minimum_records (1)
  {
    const char *capture_dir = std::getenv ("KLAYOUT_EDGE_REPLAY_CAPTURE_DIR");
    if (capture_dir) {
      directory = capture_dir;
    }

    const char *minimum = std::getenv ("KLAYOUT_EDGE_REPLAY_CAPTURE_MIN_RECORDS");
    if (minimum && *minimum) {
      char *end = 0;
      errno = 0;
      unsigned long long value = std::strtoull (minimum, &end, 10);
      if (errno == 0 && end != minimum && *end == '\0') {
        minimum_records = value > 0 ? uint64_t (value) : uint64_t (1);
      }
    }
  }

  std::string directory;
  uint64_t minimum_records;
};

const EdgeReplayCaptureConfig &
edge_replay_capture_config ()
{
  //  Environment parsing happens once, outside the scanner callback.  With no
  //  capture directory the normal path has only this guarded-static check.
  static const EdgeReplayCaptureConfig config;
  return config;
}

struct EdgeReplayHeader
{
  char magic [8];
  uint32_t version;
  uint32_t header_size;
  uint32_t record_size;
  uint32_t pair_size;
  uint32_t coordinate_bits;
  uint32_t property_bits;
  uint32_t capture_flags;
  uint32_t relation;
  uint32_t metrics;
  uint32_t zero_distance_mode;
  uint32_t ignore_angle_millidegrees;
  uint32_t option_flags;
  uint32_t process_id;
  uint32_t reserved;
  int64_t distance;
  int64_t min_projection;
  int64_t max_projection;  //  -1 is the KEDGER1 unbounded-projection sentinel
  uint64_t request_id;
  uint64_t thread_tag;
  uint64_t scanner_elapsed_ns;
  uint64_t record_count;
  uint64_t scanner_callbacks;
  uint64_t finish_callbacks;
  uint64_t unresolved_callbacks;
  uint64_t broad_pair_count;
  uint64_t exact_accept_callbacks;
  uint64_t exact_pair_count;
  uint64_t records_offset;
  uint64_t broad_pairs_offset;
  uint64_t exact_pairs_offset;
};

struct alignas (16) EdgeReplayRecord
{
  int64_t left;
  int64_t bottom;
  int64_t right;
  int64_t top;
  int64_t x1;
  int64_t y1;
  int64_t x2;
  int64_t y2;
  uint64_t property;
  uint32_t id;
  uint32_t context;
  uint32_t flags;
  uint32_t reserved;
};

struct EdgeReplayPair
{
  uint32_t first;
  uint32_t second;
};

static_assert (sizeof (size_t) <= sizeof (uint64_t), "edge replay properties require at most 64 bits");
static_assert (sizeof (EdgeReplayHeader) == 192, "unexpected edge replay header padding");
static_assert (sizeof (EdgeReplayRecord) == 96, "unexpected edge replay record padding");
static_assert (sizeof (EdgeReplayPair) == 8, "unexpected edge replay pair padding");

uint32_t
edge_replay_process_id ()
{
#if defined(_WIN32)
  return uint32_t (GetCurrentProcessId ());
#else
  return uint32_t (getpid ());
#endif
}

uint64_t
edge_replay_capture_time_ns ()
{
  return uint64_t (std::chrono::duration_cast<std::chrono::nanoseconds> (
    std::chrono::system_clock::now ().time_since_epoch ()).count ());
}

uint64_t
edge_replay_request_id ()
{
  //  This atomic is touched once per captured request, never by a scanner
  //  callback.  The callback itself writes only to request-local storage.
  static std::atomic<uint64_t> next_request (0);
  return next_request.fetch_add (1, std::memory_order_relaxed);
}

std::vector<EdgeReplayPair>
edge_replay_pairs (std::vector<uint64_t> &keys)
{
  std::sort (keys.begin (), keys.end ());
  keys.erase (std::unique (keys.begin (), keys.end ()), keys.end ());

  std::vector<EdgeReplayPair> pairs;
  pairs.reserve (keys.size ());
  for (std::vector<uint64_t>::const_iterator k = keys.begin (); k != keys.end (); ++k) {
    EdgeReplayPair pair = {
      uint32_t (*k >> 32),
      uint32_t (*k & uint64_t (std::numeric_limits<uint32_t>::max ()))
    };
    pairs.push_back (pair);
  }
  return pairs;
}

class EdgeReplayCaptureReceiver
  : public db::box_scanner_receiver<db::Edge, size_t>
{
public:
  EdgeReplayCaptureReceiver (Edge2EdgeCheckBase &output, size_t expected_records)
    : mp_output (&output), m_expected_records (expected_records),
      m_scanner_callbacks (0), m_finish_callbacks (0),
      m_unresolved_callbacks (0), m_exact_accept_callbacks (0),
      m_scanner_elapsed_ns (0)
  {
    m_records.reserve (expected_records);
    m_ids.reserve (expected_records);
  }

  void add (const db::Edge *o1, size_t p1, const db::Edge *o2, size_t p2) override
  {
    ++m_scanner_callbacks;

    uint32_t id1 = record_id (o1, p1);
    uint32_t id2 = record_id (o2, p2);
    if (id1 == 0 || id2 == 0) {
      ++m_unresolved_callbacks;
    } else {
      //  Keep the raw callback count above, but define the broad replay oracle
      //  at the same seam as Edge2EdgeCheckBase::add: after its cheap
      //  property/layer gate and immediately before the exact edge predicate.
      //  This is also the bipartite work that a replay backend must reproduce.
      if (edges_considered (mp_output->different_polygons (),
                            mp_output->requires_different_layers (), p1, p2)) {
        uint32_t first = std::min (id1, id2);
        uint32_t second = std::max (id1, id2);
        uint64_t key = (uint64_t (first) << 32) | uint64_t (second);
        m_broad_pair_keys.push_back (key);

        if (mp_output->edge_replay_capture_exact_accepts (*o1, p1, *o2, p2)) {
          ++m_exact_accept_callbacks;
          m_exact_pair_keys.push_back (key);
        }
      }
    }

    mp_output->add (o1, p1, o2, p2);
  }

  void finish (const db::Edge *edge, size_t property) override
  {
    ++m_finish_callbacks;
    if (record_id (edge, property) == 0) {
      ++m_unresolved_callbacks;
    }
    mp_output->finish (edge, property);
  }

  bool stop () const override
  {
    return mp_output->stop ();
  }

  void initialize () override
  {
    mp_output->initialize ();
  }

  void finalize (bool completed) override
  {
    mp_output->finalize (completed);
  }

  void set_scanner_elapsed_ns (uint64_t elapsed_ns)
  {
    m_scanner_elapsed_ns = elapsed_ns;
  }

  void write ()
  {
    try {
      write_internal ();
    } catch (...) {
      //  Capture is diagnostic and must never turn an otherwise valid DRC run
      //  into a failure.  An incomplete temporary file is removed below when
      //  the stream itself reports an error.
    }
  }

private:
  uint32_t record_id (const db::Edge *edge, size_t property)
  {
    if (! edge) {
      return 0;
    }

    std::unordered_map<const db::Edge *, uint32_t>::const_iterator existing = m_ids.find (edge);
    if (existing != m_ids.end ()) {
      return existing->second;
    }

    uint32_t id = uint32_t (m_records.size () + 1);
    const db::Box box = edge->bbox ();
    EdgeReplayRecord record = {
      int64_t (box.left ()), int64_t (box.bottom ()),
      int64_t (box.right ()), int64_t (box.top ()),
      int64_t (edge->p1 ().x ()), int64_t (edge->p1 ().y ()),
      int64_t (edge->p2 ().x ()), int64_t (edge->p2 ().y ()),
      uint64_t (property), id, 0,
      uint32_t (EdgeReplayHasEndpoints | ((property & size_t (1)) ? EdgeReplaySideB : 0)),
      0
    };
    m_records.push_back (record);
    m_ids.insert (std::make_pair (edge, id));
    return id;
  }

  void write_internal ()
  {
    const EdgeReplayCaptureConfig &config = edge_replay_capture_config ();
    if (config.directory.empty () || m_records.size () != m_expected_records) {
      return;
    }

    if (! tl::file_exists (config.directory) &&
        ! tl::mkpath (config.directory) &&
        ! tl::file_exists (config.directory)) {
      return;
    }

    std::vector<EdgeReplayPair> broad_pairs = edge_replay_pairs (m_broad_pair_keys);
    std::vector<EdgeReplayPair> exact_pairs = edge_replay_pairs (m_exact_pair_keys);

    const uint64_t request_id = edge_replay_request_id ();
    const uint64_t thread_tag = uint64_t (std::hash<std::thread::id> () (std::this_thread::get_id ()));
    const uint64_t capture_time_ns = edge_replay_capture_time_ns ();
    const uint32_t process_id = edge_replay_process_id ();

    std::ostringstream basename;
    basename << "edge-replay-p" << process_id
             << "-t" << std::hex << thread_tag << std::dec
             << "-r" << request_id
             << "-" << capture_time_ns << ".ker";

    const std::string final_path = tl::combine_path (config.directory, basename.str ());
    const std::string temporary_path = final_path + ".tmp";

    EdgeReplayHeader header = { };
    const char magic [8] = { 'K', 'E', 'D', 'G', 'E', 'R', '1', '\0' };
    std::copy (magic, magic + sizeof (magic), header.magic);
    header.version = 1;
    header.header_size = uint32_t (sizeof (header));
    header.record_size = uint32_t (sizeof (EdgeReplayRecord));
    header.pair_size = uint32_t (sizeof (EdgeReplayPair));
    header.coordinate_bits = 64;
    header.property_bits = 64;
    header.capture_flags = EdgeReplayBroadPairsSortedUnique |
                           EdgeReplayExactPairsSortedUnique |
                           EdgeReplayPropertyIsUint64 |
                           EdgeReplayNegativeOneIsInfiniteProjection;
    header.relation = uint32_t (db::OverlapRelation);
    header.metrics = uint32_t (db::Projection);
    header.zero_distance_mode = uint32_t (db::IncludeZeroDistanceWhenTouching);
    header.ignore_angle_millidegrees = 90000;
    header.option_flags = EdgeReplayDifferentPolygons |
                          EdgeReplayDifferentLayers |
                          EdgeReplayShielded |
                          EdgeReplayPositiveOutput;
    header.process_id = process_id;
    header.distance = int64_t (mp_output->distance ());
    header.min_projection = 0;
    //  EdgeRelationFilter::distance_type is unsigned on supported builds.
    //  KEDGER1 uses signed -1 as its explicit unbounded-projection sentinel.
    header.max_projection = -1;
    header.request_id = request_id;
    header.thread_tag = thread_tag;
    header.scanner_elapsed_ns = m_scanner_elapsed_ns;
    header.record_count = uint64_t (m_records.size ());
    header.scanner_callbacks = m_scanner_callbacks;
    header.finish_callbacks = m_finish_callbacks;
    header.unresolved_callbacks = m_unresolved_callbacks;
    header.broad_pair_count = uint64_t (broad_pairs.size ());
    header.exact_accept_callbacks = m_exact_accept_callbacks;
    header.exact_pair_count = uint64_t (exact_pairs.size ());
    header.records_offset = sizeof (header);
    header.broad_pairs_offset = header.records_offset + header.record_count * sizeof (EdgeReplayRecord);
    header.exact_pairs_offset = header.broad_pairs_offset + header.broad_pair_count * sizeof (EdgeReplayPair);

    std::ofstream stream (temporary_path.c_str (), std::ios::binary | std::ios::trunc);
    if (! stream) {
      return;
    }

    stream.write (reinterpret_cast<const char *> (&header), sizeof (header));
    if (! m_records.empty ()) {
      stream.write (reinterpret_cast<const char *> (&m_records.front ()),
                    std::streamsize (m_records.size () * sizeof (EdgeReplayRecord)));
    }
    if (! broad_pairs.empty ()) {
      stream.write (reinterpret_cast<const char *> (&broad_pairs.front ()),
                    std::streamsize (broad_pairs.size () * sizeof (EdgeReplayPair)));
    }
    if (! exact_pairs.empty ()) {
      stream.write (reinterpret_cast<const char *> (&exact_pairs.front ()),
                    std::streamsize (exact_pairs.size () * sizeof (EdgeReplayPair)));
    }
    stream.close ();

    if (! stream || ! tl::rename_file (temporary_path, final_path)) {
      tl::rm_file (temporary_path);
    }
  }

  Edge2EdgeCheckBase *mp_output;
  size_t m_expected_records;
  std::vector<EdgeReplayRecord> m_records;
  std::unordered_map<const db::Edge *, uint32_t> m_ids;
  std::vector<uint64_t> m_broad_pair_keys;
  std::vector<uint64_t> m_exact_pair_keys;
  uint64_t m_scanner_callbacks;
  uint64_t m_finish_callbacks;
  uint64_t m_unresolved_callbacks;
  uint64_t m_exact_accept_callbacks;
  uint64_t m_scanner_elapsed_ns;
};

bool
process_edge_replay_capture (Edge2EdgeCheckBase &output,
                             db::box_scanner<db::Edge, size_t> &scanner,
                             size_t expected_records)
{
  const EdgeReplayCaptureConfig &config = edge_replay_capture_config ();
  if (! output.edge_replay_capture_eligible () ||
      uint64_t (expected_records) < config.minimum_records ||
      uint64_t (expected_records) > uint64_t (std::numeric_limits<uint32_t>::max ())) {
    return false;
  }

  //  Keep the large diagnostic receiver out of poly2poly_check::process.
  //  Otherwise its cold-path stack frame and register saves penalize every
  //  production scanner request even when capture is disabled.
  EdgeReplayCaptureReceiver capture (output, expected_records);
  std::chrono::steady_clock::time_point scanner_begin = std::chrono::steady_clock::now ();
  scanner.process (capture, output.distance (), db::box_convert<db::Edge> ());
  capture.set_scanner_elapsed_ns (uint64_t (
    std::chrono::duration_cast<std::chrono::nanoseconds> (
      std::chrono::steady_clock::now () - scanner_begin).count ()));
  capture.write ();
  return true;
}

}

template <class PolygonType>
poly2poly_check<PolygonType>::poly2poly_check (Edge2EdgeCheckBase &output)
  : mp_output (& output)
{
  //  .. nothing yet ..
}

template <class PolygonType>
poly2poly_check<PolygonType>::poly2poly_check ()
  : mp_output (0)
{
  //  .. nothing yet ..
}

static size_t vertices (const db::Polygon &p)
{
  return p.vertices ();
}

static size_t vertices (const db::PolygonRef &p)
{
  return p.obj ().vertices ();
}

template <class PolygonType>
void
poly2poly_check<PolygonType>::single (const PolygonType &o, size_t p)
{
  tl_assert (! mp_output->requires_different_layers () && ! mp_output->different_polygons ());

  //  finally we check the polygons vs. itself for checks involving intra-polygon interactions

  m_scanner.clear ();
  m_scanner.reserve (vertices (o));

  m_edge_heap.clear ();

  for (typename PolygonType::polygon_edge_iterator e = o.begin_edge (); ! e.at_end (); ++e) {
    m_edge_heap.push_back (*e);
    m_scanner.insert (& m_edge_heap.back (), p);
  }

  mp_output->feed_pseudo_edges (m_scanner);

  m_scanner.process (*mp_output, mp_output->distance (), db::box_convert<db::Edge> ());
}

template <class PolygonType>
void
poly2poly_check<PolygonType>::connect (Edge2EdgeCheckBase &output)
{
  mp_output = &output;
  clear ();
}

template <class PolygonType>
void
poly2poly_check<PolygonType>::clear ()
{
  m_scanner.clear ();
  m_edge_heap.clear ();
}

template <class PolygonType>
void
poly2poly_check<PolygonType>::enter (const PolygonType &o, size_t p)
{
  for (typename PolygonType::polygon_edge_iterator e = o.begin_edge (); ! e.at_end (); ++e) {
    if (! (*e).is_degenerate ()) {
      m_edge_heap.push_back (*e);
      m_scanner.insert (& m_edge_heap.back (), p);
    }
  }
}

template <class PolygonType>
void
poly2poly_check<PolygonType>::enter (const poly2poly_check<PolygonType>::edge_type &e, size_t p)
{
  m_edge_heap.push_back (e);
  m_scanner.insert (& m_edge_heap.back (), p);
}

//  TODO: move to generic header
static bool interact (const db::Box &box, const db::Edge &e)
{
  if (! e.bbox ().touches (box)) {
    return false;
  } else if (e.is_ortho ()) {
    return true;
  } else {
    return e.clipped (box).first;
  }
}

template <class PolygonType>
void
poly2poly_check<PolygonType>::enter (const PolygonType &o, size_t p, const poly2poly_check<PolygonType>::box_type &box)
{
  if (box.empty ()) {
    return;
  }

  for (typename PolygonType::polygon_edge_iterator e = o.begin_edge (); ! e.at_end (); ++e) {
    if (! (*e).is_degenerate () && interact (box, *e)) {
      m_edge_heap.push_back (*e);
      m_scanner.insert (& m_edge_heap.back (), p);
    }
  }
}

template <class PolygonType>
void
poly2poly_check<PolygonType>::enter (const poly2poly_check<PolygonType>::edge_type &e, size_t p, const poly2poly_check<PolygonType>::box_type &box)
{
  if (! box.empty () && interact (box, e)) {
    m_edge_heap.push_back (e);
    m_scanner.insert (& m_edge_heap.back (), p);
  }
}

template <class PolygonType>
void
poly2poly_check<PolygonType>::process ()
{
  mp_output->feed_pseudo_edges (m_scanner);

  if (! edge_replay_capture_config ().directory.empty () &&
      process_edge_replay_capture (*mp_output, m_scanner, m_edge_heap.size ())) {
    return;
  }

  m_scanner.process (*mp_output, mp_output->distance (), db::box_convert<db::Edge> ());
}

//  explicit instantiations
template class poly2poly_check<db::Polygon>;
template class poly2poly_check<db::PolygonRef>;
template class poly2poly_check<db::PolygonWithProperties>;
template class poly2poly_check<db::PolygonRefWithProperties>;

// -------------------------------------------------------------------------------------
//  RegionToEdgeInteractionFilterBase implementation

template <class PolygonType, class EdgeType, class OutputType>
region_to_edge_interaction_filter_base<PolygonType, EdgeType, OutputType>::region_to_edge_interaction_filter_base (bool inverse, bool get_all)
  : m_inverse (inverse), m_get_all (get_all)
{
  //  .. nothing yet ..
}

template <class PolygonType, class EdgeType, class OutputType>
void
region_to_edge_interaction_filter_base<PolygonType, EdgeType, OutputType>::preset (const OutputType *s)
{
  m_seen.insert (s);
}

template <class PolygonType, class EdgeType, class OutputType>
void
region_to_edge_interaction_filter_base<PolygonType, EdgeType, OutputType>::add (const PolygonType *p, size_t, const EdgeType *e, size_t)
{
  const OutputType *o = 0;
  tl::select (o, p, e);

  if (m_get_all || (m_seen.find (o) == m_seen.end ()) != m_inverse) {

    //  A polygon and an edge interact if the edge is either inside completely
    //  of at least one edge of the polygon intersects with the edge
    bool interacts = false;
    if (p->box ().contains (e->p1 ()) && db::inside_poly (p->begin_edge (), e->p1 ()) >= 0) {
      interacts = true;
    } else {
      for (typename PolygonType::polygon_edge_iterator pe = p->begin_edge (); ! pe.at_end () && ! interacts; ++pe) {
        if ((*pe).intersect (*e)) {
          interacts = true;
        }
      }
    }

    if (interacts) {
      if (m_inverse) {
        m_seen.erase (o);
      } else {
        if (! m_get_all) {
          m_seen.insert (o);
        }
        put (*o);
      }
    }

  }
}

template <class PolygonType, class EdgeType, class OutputType>
void
region_to_edge_interaction_filter_base<PolygonType, EdgeType, OutputType>::fill_output ()
{
  for (typename std::set<const OutputType *>::const_iterator s = m_seen.begin (); s != m_seen.end (); ++s) {
    put (**s);
  }
}

//  explicit instantiations
template class region_to_edge_interaction_filter_base<db::Polygon, db::Edge, db::Polygon>;
template class region_to_edge_interaction_filter_base<db::PolygonRef, db::Edge, db::PolygonRef>;
template class region_to_edge_interaction_filter_base<db::Polygon, db::Edge, db::Edge>;
template class region_to_edge_interaction_filter_base<db::PolygonRef, db::Edge, db::Edge>;
template class region_to_edge_interaction_filter_base<db::PolygonWithProperties, db::EdgeWithProperties, db::PolygonWithProperties>;
template class region_to_edge_interaction_filter_base<db::PolygonRefWithProperties, db::EdgeWithProperties, db::PolygonRefWithProperties>;
template class region_to_edge_interaction_filter_base<db::PolygonWithProperties, db::EdgeWithProperties, db::EdgeWithProperties>;
template class region_to_edge_interaction_filter_base<db::PolygonRefWithProperties, db::EdgeWithProperties, db::EdgeWithProperties>;

// -------------------------------------------------------------------------------------
//  RegionToTextInteractionFilterBase implementation

template <class PolygonType, class TextType, class OutputType>
region_to_text_interaction_filter_base<PolygonType, TextType, OutputType>::region_to_text_interaction_filter_base (bool inverse, bool get_all)
  : m_inverse (inverse), m_get_all (get_all)
{
  //  .. nothing yet ..
}

template <class PolygonType, class TextType, class OutputType>
void
region_to_text_interaction_filter_base<PolygonType, TextType, OutputType>::preset (const OutputType *s)
{
  m_seen.insert (s);
}

template <class PolygonType, class TextType, class OutputType>
void
region_to_text_interaction_filter_base<PolygonType, TextType, OutputType>::add (const PolygonType *p, size_t, const TextType *t, size_t)
{
  const OutputType *o = 0;
  tl::select (o, p, t);

  if (m_get_all || (m_seen.find (o) == m_seen.end ()) != m_inverse) {

    //  A polygon and an text interact if the text is either inside completely
    //  of at least one text of the polygon intersects with the text
    db::Point pt = db::box_convert<TextType> () (*t).p1 ();
    if (p->box ().contains (pt) && db::inside_poly (p->begin_edge (), pt) >= 0) {
      if (m_inverse) {
        m_seen.erase (o);
      } else {
        if (! m_get_all) {
          m_seen.insert (o);
        }
        put (*o);
      }
    }

  }
}

template <class PolygonType, class TextType, class OutputType>
void
region_to_text_interaction_filter_base<PolygonType, TextType, OutputType>::fill_output ()
{
  for (typename std::set<const OutputType *>::const_iterator s = m_seen.begin (); s != m_seen.end (); ++s) {
    put (**s);
  }
}

//  explicit instantiations
template class region_to_text_interaction_filter_base<db::PolygonRef, db::TextRef, db::PolygonRef>;
template class region_to_text_interaction_filter_base<db::Polygon, db::Text, db::Polygon>;
template class region_to_text_interaction_filter_base<db::Polygon, db::Text, db::Text>;
template class region_to_text_interaction_filter_base<db::Polygon, db::TextRef, db::TextRef>;
template class region_to_text_interaction_filter_base<db::PolygonRef, db::TextRef, db::TextRef>;
template class region_to_text_interaction_filter_base<db::PolygonRefWithProperties, db::TextRefWithProperties, db::PolygonRefWithProperties>;
template class region_to_text_interaction_filter_base<db::PolygonWithProperties, db::TextWithProperties, db::PolygonWithProperties>;
template class region_to_text_interaction_filter_base<db::PolygonWithProperties, db::TextWithProperties, db::TextWithProperties>;
template class region_to_text_interaction_filter_base<db::PolygonWithProperties, db::TextRefWithProperties, db::TextRefWithProperties>;
template class region_to_text_interaction_filter_base<db::PolygonRefWithProperties, db::TextRefWithProperties, db::TextRefWithProperties>;

}
