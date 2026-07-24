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

} // namespace cuda_manhattan_contour

} // namespace db

#endif
