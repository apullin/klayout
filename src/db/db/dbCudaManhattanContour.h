/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaManhattanContour
#define HDR_dbCudaManhattanContour

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace db
{

namespace cuda_manhattan_contour
{

/**
 * Failure classes for the fail-closed Manhattan contour validator.
 */
enum class ValidationResult
{
  Valid,
  TooFewEdges,
  DegenerateOrNonManhattanEdge,
  OpenContour,
  SelfIntersection
};

namespace detail
{

struct Interval
{
  int64_t fixed;
  int64_t low;
  int64_t high;
  size_t edge;
};

inline bool interval_less (const Interval &a, const Interval &b)
{
  if (a.fixed != b.fixed) {
    return a.fixed < b.fixed;
  }
  if (a.low != b.low) {
    return a.low < b.low;
  }
  if (a.high != b.high) {
    return a.high < b.high;
  }
  return a.edge < b.edge;
}

inline bool adjacent (size_t a, size_t b, size_t count)
{
  return a + 1 == b || b + 1 == a ||
         (a == 0 && b + 1 == count) ||
         (b == 0 && a + 1 == count);
}

inline bool collinear_intersections_are_valid (
  std::vector<Interval> &intervals, size_t edge_count)
{
  std::sort (intervals.begin (), intervals.end (), interval_less);
  size_t group_begin = 0;
  while (group_begin < intervals.size ()) {
    size_t group_end = group_begin + 1;
    while (group_end < intervals.size () &&
           intervals [group_end].fixed == intervals [group_begin].fixed) {
      ++group_end;
    }

    int64_t furthest = intervals [group_begin].high;
    size_t furthest_edge = intervals [group_begin].edge;
    for (size_t i = group_begin + 1; i < group_end; ++i) {
      const Interval &current = intervals [i];
      if (current.low < furthest) {
        //  A positive-length collinear overlap is invalid even between
        //  neighboring contour edges.
        return false;
      }
      if (current.low == furthest &&
          ! adjacent (current.edge, furthest_edge, edge_count)) {
        //  Neighboring collinear edges may share their contour vertex.
        //  Every other point contact is a self-touch.
        return false;
      }
      if (current.high > furthest) {
        furthest = current.high;
        furthest_edge = current.edge;
      }
    }
    group_begin = group_end;
  }
  return true;
}

class FenwickCounts
{
public:
  explicit FenwickCounts (size_t size)
    : m_counts (size + 1, 0)
  {
    //  nothing else
  }

  void add (size_t index, bool insert)
  {
    for (size_t i = index + 1; i < m_counts.size (); i += i & -i) {
      if (insert) {
        ++m_counts [i];
      } else {
        --m_counts [i];
      }
    }
  }

  size_t prefix (size_t count) const
  {
    size_t result = 0;
    for (size_t i = count; i != 0; i -= i & -i) {
      result += m_counts [i];
    }
    return result;
  }

private:
  std::vector<size_t> m_counts;
};

struct SweepEvent
{
  enum Kind
  {
    AddHorizontal,
    QueryVertical,
    RemoveHorizontal
  };

  int64_t x;
  Kind kind;
  int64_t low;
  int64_t high;
};

inline bool event_less (const SweepEvent &a, const SweepEvent &b)
{
  if (a.x != b.x) {
    return a.x < b.x;
  }
  if (a.kind != b.kind) {
    //  Closed segment endpoints participate in intersection tests.
    return a.kind < b.kind;
  }
  if (a.low != b.low) {
    return a.low < b.low;
  }
  return a.high < b.high;
}

inline bool cross_intersections_are_valid (
  std::vector<SweepEvent> &events, std::vector<int64_t> &horizontal_y,
  size_t expected_intersections)
{
  std::sort (
    horizontal_y.begin (), horizontal_y.end ());
  horizontal_y.erase (
    std::unique (horizontal_y.begin (), horizontal_y.end ()),
    horizontal_y.end ());
  std::sort (events.begin (), events.end (), event_less);

  FenwickCounts active (horizontal_y.size ());
  size_t intersections = 0;
  for (std::vector<SweepEvent>::const_iterator event = events.begin ();
       event != events.end (); ++event) {
    if (event->kind == SweepEvent::AddHorizontal ||
        event->kind == SweepEvent::RemoveHorizontal) {
      const size_t y = size_t (
        std::lower_bound (
          horizontal_y.begin (), horizontal_y.end (), event->low) -
        horizontal_y.begin ());
      active.add (y, event->kind == SweepEvent::AddHorizontal);
      continue;
    }

    const size_t lower = size_t (
      std::lower_bound (
        horizontal_y.begin (), horizontal_y.end (), event->low) -
      horizontal_y.begin ());
    const size_t upper = size_t (
      std::upper_bound (
        horizontal_y.begin (), horizontal_y.end (), event->high) -
      horizontal_y.begin ());
    const size_t found = active.prefix (upper) - active.prefix (lower);
    if (intersections > expected_intersections ||
        found > expected_intersections - intersections) {
      return false;
    }
    intersections += found;
  }
  return intersections == expected_intersections;
}

} // namespace detail

/**
 * Validate a directed, closed, simple Manhattan contour in O(E log E).
 *
 * This has the same accepted topology as the former all-edge-pairs test:
 * nondegenerate Manhattan edges; exact directed closure; unique contour
 * vertices; no positive collinear overlap; and no intersection other than
 * the shared endpoint of adjacent edges.  Direction and area are deliberately
 * left to the caller because the M1/M2 scene contract checks them while
 * computing the serialized polygon bounds.
 *
 * Edge must expose signed integer x1, y1, x2 and y2 fields.
 */
template <class Edge>
ValidationResult validate (const std::vector<Edge> &edges)
{
  const size_t count = edges.size ();
  if (count < 4) {
    return ValidationResult::TooFewEdges;
  }

  std::vector<detail::Interval> horizontal;
  std::vector<detail::Interval> vertical;
  std::vector<detail::SweepEvent> events;
  std::vector<int64_t> horizontal_y;
  horizontal.reserve (count / 2 + 1);
  vertical.reserve (count / 2 + 1);
  events.reserve (count + count / 2 + 1);
  horizontal_y.reserve (count / 2 + 1);

  size_t expected_cross_intersections = 0;
  bool previous_horizontal = false;
  bool first_horizontal = false;
  for (size_t i = 0; i < count; ++i) {
    const Edge &edge = edges [i];
    const bool is_horizontal =
      edge.y1 == edge.y2 && edge.x1 != edge.x2;
    const bool is_vertical =
      edge.x1 == edge.x2 && edge.y1 != edge.y2;
    if (! (is_horizontal || is_vertical)) {
      return ValidationResult::DegenerateOrNonManhattanEdge;
    }
    const Edge &following = edges [(i + 1) % count];
    if (edge.x2 != following.x1 || edge.y2 != following.y1) {
      return ValidationResult::OpenContour;
    }

    if (i == 0) {
      first_horizontal = is_horizontal;
    } else if (is_horizontal != previous_horizontal) {
      ++expected_cross_intersections;
    }
    previous_horizontal = is_horizontal;

    if (is_horizontal) {
      const int64_t low = std::min (edge.x1, edge.x2);
      const int64_t high = std::max (edge.x1, edge.x2);
      horizontal.push_back (
        detail::Interval { edge.y1, low, high, i });
      horizontal_y.push_back (edge.y1);
      events.push_back (
        detail::SweepEvent {
          low, detail::SweepEvent::AddHorizontal, edge.y1, edge.y1 });
      events.push_back (
        detail::SweepEvent {
          high, detail::SweepEvent::RemoveHorizontal, edge.y1, edge.y1 });
    } else {
      const int64_t low = std::min (edge.y1, edge.y2);
      const int64_t high = std::max (edge.y1, edge.y2);
      vertical.push_back (
        detail::Interval { edge.x1, low, high, i });
      events.push_back (
        detail::SweepEvent {
          edge.x1, detail::SweepEvent::QueryVertical, low, high });
    }
  }
  if (previous_horizontal != first_horizontal) {
    ++expected_cross_intersections;
  }

  //  A repeated vertex necessarily creates at least one non-adjacent
  //  same-axis or cross-axis segment intersection.  The two sweeps below
  //  therefore reject it without a separate vertex tree or vertex sort.
  if (! detail::collinear_intersections_are_valid (horizontal, count) ||
      ! detail::collinear_intersections_are_valid (vertical, count) ||
      ! detail::cross_intersections_are_valid (
        events, horizontal_y, expected_cross_intersections)) {
    return ValidationResult::SelfIntersection;
  }
  return ValidationResult::Valid;
}

/**
 * Hard limits for the exact translation-equivalence cache.
 *
 * The edge limit bounds owned representative storage.  The per-key limit
 * bounds work even if an adversarial input produces many identical hashes.
 */
struct TranslationCacheLimits
{
  TranslationCacheLimits ()
    : max_representatives (4096),
      max_cached_edges (1048576),
      max_representatives_per_key (8)
  {
    //  nothing else
  }

  size_t max_representatives;
  size_t max_cached_edges;
  size_t max_representatives_per_key;
};

struct TranslationCacheStatistics
{
  TranslationCacheStatistics ()
    : lookups (0), cache_hits (0), full_validations (0),
      cached_representatives (0), cached_edges (0),
      exact_comparison_misses (0), capacity_bypasses (0)
  {
    //  nothing else
  }

  size_t lookups;
  size_t cache_hits;
  size_t full_validations;
  size_t cached_representatives;
  size_t cached_edges;
  size_t exact_comparison_misses;
  size_t capacity_bypasses;
};

namespace detail
{

struct TranslationCacheKey
{
  size_t edge_count;
  uint64_t hash_a;
  uint64_t hash_b;
};

inline bool operator< (
  const TranslationCacheKey &a, const TranslationCacheKey &b)
{
  if (a.edge_count != b.edge_count) {
    return a.edge_count < b.edge_count;
  }
  if (a.hash_a != b.hash_a) {
    return a.hash_a < b.hash_a;
  }
  return a.hash_b < b.hash_b;
}

struct ExactDelta
{
  uint64_t magnitude;
  bool negative;
};

inline ExactDelta exact_delta (int64_t value, int64_t origin)
{
  if (value >= origin) {
    return ExactDelta {
      uint64_t (value) - uint64_t (origin), false
    };
  }
  return ExactDelta {
    uint64_t (origin) - uint64_t (value), true
  };
}

inline bool operator== (const ExactDelta &a, const ExactDelta &b)
{
  return a.magnitude == b.magnitude && a.negative == b.negative;
}

template <unsigned HashBits>
struct TranslationHashMask
{
  static uint64_t value ()
  {
    return (UINT64_C (1) << HashBits) - 1;
  }
};

template <>
struct TranslationHashMask<64>
{
  static uint64_t value ()
  {
    return ~UINT64_C (0);
  }
};

inline void hash_byte (
  uint64_t &hash_a, uint64_t &hash_b, uint8_t byte)
{
  hash_a ^= uint64_t (byte);
  hash_a *= UINT64_C (1099511628211);
  hash_b ^= uint64_t (byte);
  hash_b *= UINT64_C (14029467366897019727);
}

inline void hash_delta (
  uint64_t &hash_a, uint64_t &hash_b, const ExactDelta &delta)
{
  hash_byte (hash_a, hash_b, delta.negative ? 1 : 0);
  for (unsigned shift = 0; shift < 64; shift += 8) {
    hash_byte (
      hash_a, hash_b, uint8_t (delta.magnitude >> shift));
  }
}

template <class Edge>
TranslationCacheKey translation_cache_key (
  const std::vector<Edge> &edges)
{
  uint64_t hash_a = UINT64_C (1469598103934665603);
  uint64_t hash_b = UINT64_C (7809847782465536322);
  const int64_t origin_x = edges.front ().x1;
  const int64_t origin_y = edges.front ().y1;
  for (typename std::vector<Edge>::const_iterator edge = edges.begin ();
       edge != edges.end (); ++edge) {
    hash_delta (
      hash_a, hash_b, exact_delta (edge->x1, origin_x));
    hash_delta (
      hash_a, hash_b, exact_delta (edge->y1, origin_y));
    hash_delta (
      hash_a, hash_b, exact_delta (edge->x2, origin_x));
    hash_delta (
      hash_a, hash_b, exact_delta (edge->y2, origin_y));
  }
  return TranslationCacheKey { edges.size (), hash_a, hash_b };
}

template <class Edge>
bool exact_translation_equivalent (
  const std::vector<Edge> &a, const std::vector<Edge> &b)
{
  if (a.size () != b.size () || a.empty ()) {
    return false;
  }

  const int64_t a_origin_x = a.front ().x1;
  const int64_t a_origin_y = a.front ().y1;
  const int64_t b_origin_x = b.front ().x1;
  const int64_t b_origin_y = b.front ().y1;
  for (size_t i = 0; i < a.size (); ++i) {
    if (! (exact_delta (a [i].x1, a_origin_x) ==
           exact_delta (b [i].x1, b_origin_x)) ||
        ! (exact_delta (a [i].y1, a_origin_y) ==
           exact_delta (b [i].y1, b_origin_y)) ||
        ! (exact_delta (a [i].x2, a_origin_x) ==
           exact_delta (b [i].x2, b_origin_x)) ||
        ! (exact_delta (a [i].y2, a_origin_y) ==
           exact_delta (b [i].y2, b_origin_y))) {
      return false;
    }
  }
  return true;
}

} // namespace detail

/**
 * Memoize only exact translations of contours already proven Valid.
 *
 * Hashes only select a bounded candidate bucket.  Every hit is confirmed by
 * an overflow-free, coordinate-by-coordinate translation comparison.  A
 * mismatch or any capacity limit always runs the full validator, and invalid
 * contours are never cached.
 */
template <class Edge, unsigned HashBits = 64>
class TranslationValidationCache
{
public:
  explicit TranslationValidationCache (
    const TranslationCacheLimits &limits = TranslationCacheLimits ())
    : m_limits (limits), m_statistics (), m_representatives ()
  {
    //  nothing else
    static_assert (
      HashBits <= 64,
      "translation-cache hash width cannot exceed uint64");
  }

  ValidationResult validate_contour (const std::vector<Edge> &edges)
  {
    ++m_statistics.lookups;

    if (! edges.empty () &&
        edges.size () <= m_limits.max_cached_edges) {
      detail::TranslationCacheKey key =
        detail::translation_cache_key (edges);
      const uint64_t hash_mask =
        detail::TranslationHashMask<HashBits>::value ();
      key.hash_a &= hash_mask;
      key.hash_b &= hash_mask;
      typename RepresentativeMap::const_iterator bucket =
        m_representatives.find (key);
      if (bucket != m_representatives.end ()) {
        for (typename RepresentativeList::const_iterator representative =
               bucket->second.begin ();
             representative != bucket->second.end (); ++representative) {
          if (detail::exact_translation_equivalent (
                edges, *representative)) {
            ++m_statistics.cache_hits;
            return ValidationResult::Valid;
          }
          ++m_statistics.exact_comparison_misses;
        }
      }

      ++m_statistics.full_validations;
      const ValidationResult result =
        cuda_manhattan_contour::validate (edges);
      if (result != ValidationResult::Valid) {
        return result;
      }

      const size_t bucket_size =
        bucket == m_representatives.end () ? 0 : bucket->second.size ();
      if (m_statistics.cached_representatives <
            m_limits.max_representatives &&
          edges.size () <=
            m_limits.max_cached_edges - m_statistics.cached_edges &&
          bucket_size < m_limits.max_representatives_per_key) {
        m_representatives [key].push_back (edges);
        ++m_statistics.cached_representatives;
        m_statistics.cached_edges += edges.size ();
      } else {
        ++m_statistics.capacity_bypasses;
      }
      return result;
    }

    ++m_statistics.full_validations;
    const ValidationResult result =
      cuda_manhattan_contour::validate (edges);
    if (result == ValidationResult::Valid) {
      ++m_statistics.capacity_bypasses;
    }
    return result;
  }

  const TranslationCacheStatistics &statistics () const
  {
    return m_statistics;
  }

private:
  typedef std::vector<std::vector<Edge> > RepresentativeList;
  typedef std::map<
    detail::TranslationCacheKey, RepresentativeList> RepresentativeMap;

  TranslationCacheLimits m_limits;
  TranslationCacheStatistics m_statistics;
  RepresentativeMap m_representatives;
};

} // namespace cuda_manhattan_contour

} // namespace db

#endif
