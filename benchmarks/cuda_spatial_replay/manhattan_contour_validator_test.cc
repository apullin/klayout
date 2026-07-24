/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaManhattanContour.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{

struct Edge
{
  int64_t x1, y1, x2, y2;
};

typedef std::pair<int64_t, int64_t> Point;

std::vector<Edge> contour (const std::vector<Point> &points)
{
  std::vector<Edge> result;
  result.reserve (points.size ());
  for (size_t i = 0; i < points.size (); ++i) {
    const Point &a = points [i];
    const Point &b = points [(i + 1) % points.size ()];
    result.push_back (Edge { a.first, a.second, b.first, b.second });
  }
  return result;
}

bool ranges_intersect (
  int64_t a0, int64_t a1, int64_t b0, int64_t b1)
{
  return std::max (std::min (a0, a1), std::min (b0, b1)) <=
         std::min (std::max (a0, a1), std::max (b0, b1));
}

bool segments_intersect (const Edge &a, const Edge &b)
{
  const bool ah = a.y1 == a.y2;
  const bool bh = b.y1 == b.y2;
  if (ah && bh) {
    return a.y1 == b.y1 &&
           ranges_intersect (a.x1, a.x2, b.x1, b.x2);
  }
  if (! ah && ! bh) {
    return a.x1 == b.x1 &&
           ranges_intersect (a.y1, a.y2, b.y1, b.y2);
  }
  const Edge &horizontal = ah ? a : b;
  const Edge &vertical = ah ? b : a;
  return std::min (horizontal.x1, horizontal.x2) <= vertical.x1 &&
         vertical.x1 <= std::max (horizontal.x1, horizontal.x2) &&
         std::min (vertical.y1, vertical.y2) <= horizontal.y1 &&
         horizontal.y1 <= std::max (vertical.y1, vertical.y2);
}

bool positive_collinear_overlap (const Edge &a, const Edge &b)
{
  if (a.y1 == a.y2 && b.y1 == b.y2 && a.y1 == b.y1) {
    return std::min (std::max (a.x1, a.x2), std::max (b.x1, b.x2)) >
           std::max (std::min (a.x1, a.x2), std::min (b.x1, b.x2));
  }
  if (a.x1 == a.x2 && b.x1 == b.x2 && a.x1 == b.x1) {
    return std::min (std::max (a.y1, a.y2), std::max (b.y1, b.y2)) >
           std::max (std::min (a.y1, a.y2), std::min (b.y1, b.y2));
  }
  return false;
}

//  The exact former O(E^2) acceptance condition, retained only as a test
//  oracle for bounded differential cases.
bool quadratic_reference (const std::vector<Edge> &edges)
{
  if (edges.size () < 4) {
    return false;
  }
  std::set<Point> vertices;
  for (size_t i = 0; i < edges.size (); ++i) {
    const Edge &edge = edges [i];
    const bool horizontal =
      edge.y1 == edge.y2 && edge.x1 != edge.x2;
    const bool vertical =
      edge.x1 == edge.x2 && edge.y1 != edge.y2;
    if (! (horizontal || vertical) ||
        ! vertices.insert (Point (edge.x1, edge.y1)).second) {
      return false;
    }
    const Edge &following = edges [(i + 1) % edges.size ()];
    if (edge.x2 != following.x1 || edge.y2 != following.y1) {
      return false;
    }
  }
  for (size_t first = 0; first < edges.size (); ++first) {
    for (size_t second = first + 1; second < edges.size (); ++second) {
      if (! segments_intersect (edges [first], edges [second])) {
        continue;
      }
      const bool adjacent =
        second == first + 1 ||
        (first == 0 && second + 1 == edges.size ());
      if (! adjacent ||
          positive_collinear_overlap (edges [first], edges [second])) {
        return false;
      }
    }
  }
  return true;
}

bool linearithmic (const std::vector<Edge> &edges)
{
  return db::cuda_manhattan_contour::validate (edges) ==
         db::cuda_manhattan_contour::ValidationResult::Valid;
}

bool expect (
  const char *name, const std::vector<Edge> &edges, bool accepted)
{
  const bool reference = quadratic_reference (edges);
  const bool actual = linearithmic (edges);
  if (reference != accepted || actual != accepted) {
    std::cerr << name << " failed: expected=" << accepted
              << " reference=" << reference << " actual=" << actual
              << " edges=" << edges.size () << '\n';
    return false;
  }
  return true;
}

std::vector<Edge> large_staircase (size_t steps)
{
  std::vector<Point> points;
  points.reserve (steps * 2 + 4);
  points.push_back (Point (0, 0));
  points.push_back (Point (0, 1));
  for (size_t i = 1; i <= steps; ++i) {
    points.push_back (Point (int64_t (i), int64_t (i)));
    points.push_back (Point (int64_t (i), int64_t (i + 1)));
  }
  points.push_back (
    Point (int64_t (steps + 1), int64_t (steps + 1)));
  points.push_back (Point (int64_t (steps + 1), 0));
  return contour (points);
}

uint64_t load_u64 (const unsigned char *bytes)
{
  uint64_t result = 0;
  for (unsigned i = 0; i < 8; ++i) {
    result |= uint64_t (bytes [i]) << (8 * i);
  }
  return result;
}

uint32_t load_u32 (const unsigned char *bytes)
{
  uint32_t result = 0;
  for (unsigned i = 0; i < 4; ++i) {
    result |= uint32_t (bytes [i]) << (8 * i);
  }
  return result;
}

int64_t load_i64 (const unsigned char *bytes)
{
  return static_cast<int64_t> (load_u64 (bytes));
}

bool read_exact (
  std::ifstream &input, unsigned char *bytes, size_t count)
{
  if (count > size_t (std::numeric_limits<std::streamsize>::max ())) {
    return false;
  }
  input.read (
    reinterpret_cast<char *> (bytes), std::streamsize (count));
  return input.good () || size_t (input.gcount ()) == count;
}

bool production_scene_microbenchmark (const char *path)
{
  std::ifstream input (path, std::ios::in | std::ios::binary);
  unsigned char header [256] = {};
  if (! input || ! read_exact (input, header, sizeof (header)) ||
      std::memcmp (header, "KM1WSCN1", 8) != 0 ||
      load_u32 (header + 8) != 1 ||
      load_u32 (header + 12) != sizeof (header)) {
    std::cerr << "unable to read KM1WSCN1 microbenchmark header: "
              << path << '\n';
    return false;
  }

  const uint64_t polygon_offset = load_u64 (header + 72);
  const uint64_t edge_offset = load_u64 (header + 80);
  const uint64_t polygon_count = load_u64 (header + 112);
  const uint64_t edge_count = load_u64 (header + 120);
  if (polygon_count > size_t (-1) / 48 ||
      edge_count > size_t (-1) / 32 ||
      polygon_offset > uint64_t (std::numeric_limits<std::streamoff>::max ()) ||
      edge_offset > uint64_t (std::numeric_limits<std::streamoff>::max ())) {
    std::cerr << "KM1WSCN1 microbenchmark census is not host-sized\n";
    return false;
  }

  std::vector<unsigned char> raw_polygons (size_t (polygon_count) * 48);
  std::vector<unsigned char> raw_edges (size_t (edge_count) * 32);
  input.seekg (std::streamoff (polygon_offset));
  if (! read_exact (
        input, raw_polygons.data (), raw_polygons.size ())) {
    std::cerr << "unable to read KM1WSCN1 polygon records\n";
    return false;
  }
  input.seekg (std::streamoff (edge_offset));
  if (! read_exact (input, raw_edges.data (), raw_edges.size ())) {
    std::cerr << "unable to read KM1WSCN1 edge records\n";
    return false;
  }

  struct Span
  {
    uint64_t begin;
    uint32_t count;
  };
  std::vector<Span> spans;
  spans.reserve (size_t (polygon_count));
  uint64_t expected_edge = 0;
  size_t maximum_edges = 0;
  for (size_t id = 0; id < size_t (polygon_count); ++id) {
    const unsigned char *record = raw_polygons.data () + id * 48;
    const Span span = { load_u64 (record), load_u32 (record + 44) };
    if (span.begin != expected_edge || expected_edge > edge_count ||
        span.count > edge_count - expected_edge) {
      std::cerr << "KM1WSCN1 polygon spans are not canonical\n";
      return false;
    }
    expected_edge += span.count;
    maximum_edges = std::max (maximum_edges, size_t (span.count));
    spans.push_back (span);
  }
  if (expected_edge != edge_count) {
    std::cerr << "KM1WSCN1 polygon spans do not cover the edge table\n";
    return false;
  }

  const auto validate_pass = [&] (double *seconds) {
    const std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now ();
    for (size_t polygon_id = 0; polygon_id < spans.size ();
         ++polygon_id) {
      const Span &span = spans [polygon_id];
      std::vector<Edge> edges;
      edges.reserve (span.count);
      for (uint32_t local = 0; local < span.count; ++local) {
        const unsigned char *record =
          raw_edges.data () + size_t (span.begin + local) * 32;
        edges.push_back (
          Edge {
            load_i64 (record), load_i64 (record + 8),
            load_i64 (record + 16), load_i64 (record + 24)
          });
      }
      if (! linearithmic (edges)) {
        std::cerr << "KM1WSCN1 validator rejected polygon "
                  << polygon_id << " with " << span.count << " edges\n";
        return false;
      }
    }
    *seconds =
      std::chrono::duration<double> (
        std::chrono::steady_clock::now () - begin).count ();
    return true;
  };

  double first = 0.0;
  double second = 0.0;
  if (! validate_pass (&first) || ! validate_pass (&second)) {
    return false;
  }
  std::cout << "KM1WSCN1 exact contour passes: polygons="
            << polygon_count << " edges=" << edge_count
            << " maximum_polygon_edges=" << maximum_edges
            << " host_pass=" << first << " s"
            << " DSO_pass=" << second << " s"
            << " combined=" << first + second << " s\n";
  return true;
}

std::vector<Edge> random_closed_walk (std::mt19937_64 &random)
{
  std::uniform_int_distribution<int> coordinate (-8, 8);
  std::uniform_int_distribution<int> axis (0, 1);
  for (;;) {
    std::vector<Point> points;
    points.push_back (Point (0, 0));
    int64_t x = 0;
    int64_t y = 0;
    const size_t random_edges = 2 + size_t (random () % 12);
    for (size_t i = 0; i < random_edges; ++i) {
      int64_t next = 0;
      if (axis (random) == 0) {
        do {
          next = coordinate (random);
        } while (next == x);
        x = next;
      } else {
        do {
          next = coordinate (random);
        } while (next == y);
        y = next;
      }
      points.push_back (Point (x, y));
    }
    if (x == 0 || y == 0) {
      continue;
    }
    points.push_back (Point (0, y));
    return contour (points);
  }
}

} // anonymous namespace

int main (int argc, char *argv [])
{
  bool good = true;
  good = expect (
    "rectangle",
    contour ({ { 0, 0 }, { 0, 4 }, { 6, 4 }, { 6, 0 } }),
    true) && good;
  good = expect (
    "concave",
    contour ({
      { 0, 0 }, { 0, 6 }, { 2, 6 }, { 2, 3 },
      { 4, 3 }, { 4, 6 }, { 6, 6 }, { 6, 0 }
    }),
    true) && good;
  good = expect (
    "redundant-collinear-vertex",
    contour ({
      { 0, 0 }, { 0, 4 }, { 2, 4 }, { 4, 4 }, { 4, 0 }
    }),
    true) && good;
  good = expect (
    "orthogonal-crossing",
    contour ({
      { 0, 0 }, { 0, 6 }, { 4, 6 }, { 4, 2 }, { -2, 2 },
      { -2, 4 }, { 6, 4 }, { 6, 0 }
    }),
    false) && good;
  good = expect (
    "interior-touch",
    contour ({
      { 0, 0 }, { 0, 4 }, { 4, 4 }, { 4, 2 }, { 2, 2 },
      { 2, 4 }, { 2, 6 }, { 6, 6 }, { 6, 0 }
    }),
    false) && good;
  good = expect (
    "adjacent-collinear-overlap",
    contour ({
      { 0, 0 }, { 0, 4 }, { 4, 4 }, { 2, 4 }, { 2, 0 }
    }),
    false) && good;
  good = expect (
    "same-axis-nested-overlap",
    contour ({
      { 0, 0 }, { 0, 8 }, { 8, 8 }, { 8, 6 }, { 2, 6 },
      { 2, 8 }, { 6, 8 }, { 6, 4 }, { 10, 4 }, { 10, 0 }
    }),
    false) && good;
  good = expect (
    "nonadjacent-endpoint-touch",
    contour ({
      { 0, 0 }, { 0, 4 }, { -4, 4 }, { -4, 0 },
      { 0, 0 }, { 0, -4 }, { 4, -4 }, { 4, 0 }
    }),
    false) && good;
  good = expect (
    "repeated-vertex",
    contour ({
      { 0, 0 }, { 0, 4 }, { 4, 4 }, { 4, 0 },
      { 0, 0 }, { 0, -2 }, { 6, -2 }, { 6, 0 }
    }),
    false) && good;
  good = expect (
    "wraparound-collinear-adjacency",
    contour ({
      { 0, 2 }, { 0, 4 }, { 4, 4 }, { 4, 0 },
      { 0, 0 }, { 0, 1 }
    }),
    true) && good;

  std::vector<Edge> open =
    contour ({ { 0, 0 }, { 0, 4 }, { 4, 4 }, { 4, 0 } });
  open [1].x2 = 3;
  good = expect ("open-contour", open, false) && good;

  std::vector<Edge> degenerate =
    contour ({ { 0, 0 }, { 0, 4 }, { 4, 4 }, { 4, 0 } });
  degenerate [0].x2 = degenerate [0].x1;
  degenerate [0].y2 = degenerate [0].y1;
  good = expect ("degenerate-edge", degenerate, false) && good;

  std::mt19937_64 random (UINT64_C (0x4d32434f4e544f55));
  const size_t differential_cases = 50000;
  for (size_t test = 0; test < differential_cases; ++test) {
    std::vector<Edge> candidate = random_closed_walk (random);
    if ((random () & 15) == 0 && ! candidate.empty ()) {
      //  Include malformed-open records as well as closed random walks.
      candidate [random () % candidate.size ()].x2 += 1;
    }
    const bool expected = quadratic_reference (candidate);
    const bool actual = linearithmic (candidate);
    if (expected != actual) {
      std::cerr << "random differential mismatch at case " << test
                << ": reference=" << expected << " actual=" << actual
                << " edges=" << candidate.size () << '\n';
      good = false;
      break;
    }
  }

  good = expect (
    "bounded-large-staircase-reference",
    large_staircase (500), true) && good;

  const std::vector<Edge> comparative = large_staircase (5000);
  const std::chrono::steady_clock::time_point quadratic_begin =
    std::chrono::steady_clock::now ();
  const bool quadratic_valid = quadratic_reference (comparative);
  const double quadratic_seconds =
    std::chrono::duration<double> (
      std::chrono::steady_clock::now () - quadratic_begin).count ();
  const std::chrono::steady_clock::time_point linearithmic_begin =
    std::chrono::steady_clock::now ();
  const bool linearithmic_valid = linearithmic (comparative);
  const double linearithmic_seconds =
    std::chrono::duration<double> (
      std::chrono::steady_clock::now () - linearithmic_begin).count ();
  if (! quadratic_valid || ! linearithmic_valid) {
    std::cerr << "comparative staircase rejected\n";
    good = false;
  }

  const std::vector<Edge> large = large_staircase (50000);
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  const bool large_valid = linearithmic (large);
  const double seconds =
    std::chrono::duration<double> (
      std::chrono::steady_clock::now () - begin).count ();
  if (! large_valid) {
    std::cerr << "large staircase rejected\n";
    good = false;
  }

  std::cout << "Manhattan contour validator: "
            << differential_cases << " quadratic differential cases, "
            << large.size () << "-edge large contour in "
            << seconds << " s; " << comparative.size ()
            << "-edge old/new=" << quadratic_seconds << "/"
            << linearithmic_seconds << " s\n";
  if (argc == 3 && std::strcmp (argv [1], "--scene") == 0) {
    good = production_scene_microbenchmark (argv [2]) && good;
  } else if (argc != 1) {
    std::cerr << "usage: " << argv [0] << " [--scene KM1WSCN1]\n";
    good = false;
  }
  return good ? 0 : 1;
}
