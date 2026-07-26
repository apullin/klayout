
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


#include "tlUnitTest.h"
#include "dbHierNetworkProcessor.h"
#include "dbTestSupport.h"
#include "dbShapeRepository.h"
#include "dbNetShape.h"
#include "dbPolygon.h"
#include "dbPath.h"
#include "dbText.h"
#include "dbLayout.h"
#include "dbStream.h"
#include "dbCommonReader.h"
#include "tlEnv.h"

#include <limits>
#include <utility>

static std::string l2s (db::Connectivity::layer_iterator b, db::Connectivity::layer_iterator e)
{
  std::string s;
  for (db::Connectivity::layer_iterator i = b; i != e; ++i) {
    if (! s.empty ()) {
      s += ",";
    }
    s += tl::to_string (i->first);
    if (i->second < 0) {
      s += "-S";
    } else if (i->second > 0) {
      s += "+S";
    }
  }
  return s;
}

static std::string al2s (db::Connectivity::all_layer_iterator b, db::Connectivity::all_layer_iterator e)
{
  std::string s;
  for (db::Connectivity::all_layer_iterator i = b; i != e; ++i) {
    if (! s.empty ()) {
      s += ",";
    }
    s += tl::to_string (*i);
  }
  return s;
}

static std::string gn2s (db::Connectivity::global_nets_iterator b, db::Connectivity::global_nets_iterator e)
{
  std::string s;
  for (db::Connectivity::global_nets_iterator i = b; i != e; ++i) {
    if (! s.empty ()) {
      s += ",";
    }
    s += tl::to_string (i->first);
    if (i->second < 0) {
      s += "-S";
    } else if (i->second > 0) {
      s += "+S";
    }
  }
  return s;
}

TEST(1_Connectivity)
{
  db::Connectivity conn;

  EXPECT_EQ (al2s (conn.begin_layers (), conn.end_layers ()), "");

  conn.connect (0);
  EXPECT_EQ (al2s (conn.begin_layers (), conn.end_layers ()), "0");
  EXPECT_EQ (l2s (conn.begin_connected (0), conn.end_connected (0)), "0");
  EXPECT_EQ (l2s (conn.begin_connected (1), conn.end_connected (1)), "");

  conn.connect (0, 1);
  EXPECT_EQ (al2s (conn.begin_layers (), conn.end_layers ()), "0,1");
  EXPECT_EQ (l2s (conn.begin_connected (0), conn.end_connected (0)), "0,1");
  EXPECT_EQ (l2s (conn.begin_connected (1), conn.end_connected (1)), "0");

  conn.connect (1);
  EXPECT_EQ (l2s (conn.begin_connected (1), conn.end_connected (1)), "0,1");

  conn.connect (0, 2);
  conn.connect (2);

  EXPECT_EQ (l2s (conn.begin_connected (0), conn.end_connected (0)), "0,1,2");
  EXPECT_EQ (l2s (conn.begin_connected (1), conn.end_connected (1)), "0,1");
  EXPECT_EQ (l2s (conn.begin_connected (2), conn.end_connected (2)), "0,2");

  EXPECT_EQ (conn.connect_global (0, "GLOBAL"), size_t (0));
  EXPECT_EQ (gn2s (conn.begin_global_connections (2), conn.end_global_connections (2)), "");
  EXPECT_EQ (gn2s (conn.begin_global_connections (0), conn.end_global_connections (0)), "0");
  EXPECT_EQ (conn.connect_global (2, "GLOBAL2"), size_t (1));
  EXPECT_EQ (gn2s (conn.begin_global_connections (2), conn.end_global_connections (2)), "1");
  EXPECT_EQ (conn.connect_global (0, "GLOBAL2"), size_t (1));
  EXPECT_EQ (gn2s (conn.begin_global_connections (0), conn.end_global_connections (0)), "0,1");

  EXPECT_EQ (conn.global_net_name (0), "GLOBAL");
  EXPECT_EQ (conn.global_net_name (1), "GLOBAL2");

  db::Connectivity conn2 = conn;

  EXPECT_EQ (l2s (conn2.begin_connected (0), conn2.end_connected (0)), "0,1,2");
  EXPECT_EQ (l2s (conn2.begin_connected (1), conn2.end_connected (1)), "0,1");
  EXPECT_EQ (l2s (conn2.begin_connected (2), conn2.end_connected (2)), "0,2");

  EXPECT_EQ (gn2s (conn2.begin_global_connections (0), conn2.end_global_connections (0)), "0,1");
  EXPECT_EQ (conn2.global_net_name (0), "GLOBAL");
  EXPECT_EQ (conn2.global_net_name (1), "GLOBAL2");
}

TEST(1_ConnectivitySoft)
{
  db::Connectivity conn;

  EXPECT_EQ (al2s (conn.begin_layers (), conn.end_layers ()), "");

  conn.connect (0);
  EXPECT_EQ (al2s (conn.begin_layers (), conn.end_layers ()), "0");
  EXPECT_EQ (l2s (conn.begin_connected (0), conn.end_connected (0)), "0");
  EXPECT_EQ (l2s (conn.begin_connected (1), conn.end_connected (1)), "");

  conn.soft_connect (0, 1);
  EXPECT_EQ (al2s (conn.begin_layers (), conn.end_layers ()), "0,1");
  EXPECT_EQ (l2s (conn.begin_connected (0), conn.end_connected (0)), "0,1-S");
  EXPECT_EQ (l2s (conn.begin_connected (1), conn.end_connected (1)), "0+S");

  conn.connect (1);
  EXPECT_EQ (l2s (conn.begin_connected (1), conn.end_connected (1)), "0+S,1");

  conn.soft_connect (2, 0);
  conn.connect (2);

  EXPECT_EQ (l2s (conn.begin_connected (0), conn.end_connected (0)), "0,1-S,2+S");
  EXPECT_EQ (l2s (conn.begin_connected (1), conn.end_connected (1)), "0+S,1");
  EXPECT_EQ (l2s (conn.begin_connected (2), conn.end_connected (2)), "0-S,2");

  conn.connect (2, 0);

  EXPECT_EQ (l2s (conn.begin_connected (0), conn.end_connected (0)), "0,1-S,2");
  EXPECT_EQ (l2s (conn.begin_connected (1), conn.end_connected (1)), "0+S,1");
  EXPECT_EQ (l2s (conn.begin_connected (2), conn.end_connected (2)), "0,2");

  EXPECT_EQ (conn.soft_connect_global (0, "GLOBAL"), size_t (0));
  EXPECT_EQ (gn2s (conn.begin_global_connections (2), conn.end_global_connections (2)), "");
  EXPECT_EQ (gn2s (conn.begin_global_connections (0), conn.end_global_connections (0)), "0-S");
  EXPECT_EQ (conn.soft_connect_global (2, "GLOBAL2"), size_t (1));
  EXPECT_EQ (gn2s (conn.begin_global_connections (2), conn.end_global_connections (2)), "1-S");
  EXPECT_EQ (conn.connect_global (0, "GLOBAL2"), size_t (1));
  EXPECT_EQ (gn2s (conn.begin_global_connections (0), conn.end_global_connections (0)), "0-S,1");

  EXPECT_EQ (conn.global_net_name (0), "GLOBAL");
  EXPECT_EQ (conn.global_net_name (1), "GLOBAL2");

  db::Connectivity conn2 = conn;

  EXPECT_EQ (l2s (conn2.begin_connected (0), conn2.end_connected (0)), "0,1-S,2");
  EXPECT_EQ (l2s (conn2.begin_connected (1), conn2.end_connected (1)), "0+S,1");
  EXPECT_EQ (l2s (conn2.begin_connected (2), conn2.end_connected (2)), "0,2");

  EXPECT_EQ (gn2s (conn2.begin_global_connections (0), conn2.end_global_connections (0)), "0-S,1");
  EXPECT_EQ (conn2.global_net_name (0), "GLOBAL");
  EXPECT_EQ (conn2.global_net_name (1), "GLOBAL2");
}

TEST(2_ShapeInteractions)
{
  db::Connectivity conn;

  conn.connect (0);
  conn.connect (1);
  conn.connect (0, 1);

  db::Polygon poly;
  tl::from_string ("(0,0;0,1000;1000,1000;1000,0)", poly);
  db::GenericRepository repo;
  db::PolygonRef ref1 (poly, repo);
  db::ICplxTrans t2 (db::Trans (db::Vector (0, 10)));
  db::PolygonRef ref2 (poly.transformed (t2), repo);
  db::ICplxTrans t3 (db::Trans (db::Vector (0, 2000)));
  db::PolygonRef ref3 (poly.transformed (t3), repo);

  int soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (conn.interacts (ref1, 0, ref2, 0, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 0, ref1, 0, t2, soft), true);  // t2*ref1 == ref2
  EXPECT_EQ (conn.interacts (ref1, 0, ref2, 1, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 0, ref1, 1, t2, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 1, ref2, 0, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 1, ref1, 0, t2, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 0, ref3, 0, soft), false);
  EXPECT_EQ (conn.interacts (ref1, 0, ref1, 0, t3, soft), false);  // t3*ref1 == ref3
  EXPECT_EQ (conn.interacts (ref1, 0, ref3, 1, soft), false);
  EXPECT_EQ (conn.interacts (ref1, 0, ref1, 1, t3, soft), false);
  EXPECT_EQ (conn.interacts (ref1, 1, ref2, 2, soft), false);
  EXPECT_EQ (conn.interacts (ref1, 1, ref1, 2, t2, soft), false);
}

TEST(2_ShapeInteractionsRealPolygon)
{
  db::Connectivity conn;

  conn.connect (0);
  conn.connect (1);
  conn.connect (0, 1);

  db::Polygon poly;
  tl::from_string ("(0,0;0,1000;500,1000;500,1500;1000,1500;1000,0)", poly);
  db::GenericRepository repo;
  db::PolygonRef ref1 (poly, repo);
  db::ICplxTrans t2 (db::Trans (db::Vector (0, 10)));
  db::PolygonRef ref2 (poly.transformed (t2), repo);
  db::ICplxTrans t3 (db::Trans (db::Vector (0, 2000)));
  db::PolygonRef ref3 (poly.transformed (t3), repo);
  db::ICplxTrans t4 (db::Trans (db::Vector (0, 1500)));
  db::PolygonRef ref4 (poly.transformed (t4), repo);

  int soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (conn.interacts (ref1, 0, ref2, 0, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 0, ref1, 0, t2, soft), true);  // t2*ref1 == ref2
  EXPECT_EQ (conn.interacts (ref1, 0, ref2, 1, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 0, ref1, 1, t2, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 1, ref2, 0, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 1, ref1, 0, t2, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 0, ref3, 0, soft), false);
  EXPECT_EQ (conn.interacts (ref1, 0, ref1, 0, t3, soft), false);
  EXPECT_EQ (conn.interacts (ref1, 0, ref4, 0, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 0, ref1, 0, t4, soft), true);
  EXPECT_EQ (conn.interacts (ref1, 0, ref3, 1, soft), false);
  EXPECT_EQ (conn.interacts (ref1, 0, ref1, 1, t3, soft), false);
  EXPECT_EQ (conn.interacts (ref1, 1, ref2, 2, soft), false);
  EXPECT_EQ (conn.interacts (ref1, 1, ref1, 2, t2, soft), false);
}

TEST(10_LocalClusterBasic)
{
  db::GenericRepository repo;

  db::Polygon poly;
  tl::from_string ("(0,0;0,1000;1000,1000;1000,0)", poly);

  db::local_cluster<db::PolygonRef> cluster;
  EXPECT_EQ (cluster.bbox ().to_string (), "()");
  EXPECT_EQ (cluster.id (), size_t (0));

  cluster.add (db::PolygonRef (poly, repo), 0);
  cluster.add_attr (1);
  EXPECT_EQ (cluster.bbox ().to_string (), "(0,0;1000,1000)");

  db::local_cluster<db::PolygonRef> cluster2;
  cluster2.add (db::PolygonRef (poly, repo).transformed (db::Disp (db::Vector (10, 20))), 1);
  cluster2.add_attr (2);

  cluster.join_with (cluster2);
  EXPECT_EQ (cluster.bbox ().to_string (), "(0,0;1010,1020)");

  EXPECT_EQ (cluster.begin_attr () == cluster.end_attr (), false);
  db::local_cluster<db::PolygonRef>::attr_iterator a = cluster.begin_attr ();
  EXPECT_EQ (*a++, 1u);
  EXPECT_EQ (*a++, 2u);
  EXPECT_EQ (a == cluster.end_attr (), true);
}

TEST(11_LocalClusterInteractBasic)
{
  db::GenericRepository repo;

  db::Connectivity conn;
  conn.connect (0);
  conn.connect (1);
  conn.connect (2);
  conn.connect (0, 1);
  conn.connect (0, 2);

  db::Polygon poly;
  tl::from_string ("(0,0;0,1000;1000,1000;1000,0)", poly);

  db::local_cluster<db::PolygonRef> cluster;
  db::local_cluster<db::PolygonRef> cluster2;
  int soft;

  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), false);

  cluster.add (db::PolygonRef (poly, repo), 0);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), false);

  cluster2.add (db::PolygonRef (poly, repo), 0);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), true);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (db::Trans (db::Vector (10, 20))), conn, soft), true);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (db::Trans (db::Vector (0, 1000))), conn, soft), true);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (db::Trans (db::Vector (0, 1001))), conn, soft), false);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (db::Trans (db::Vector (0, 2000))), conn, soft), false);

  cluster.clear ();
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), false);
}

TEST(11_LocalClusterInteractDifferentLayers)
{
  db::GenericRepository repo;

  db::Connectivity conn;
  conn.connect (0);
  conn.connect (1);
  conn.connect (2);
  conn.connect (0, 1);
  conn.connect (0, 2);

  db::Polygon poly;
  tl::from_string ("(0,0;0,1000;1000,1000;1000,0)", poly);

  db::local_cluster<db::PolygonRef> cluster;
  db::local_cluster<db::PolygonRef> cluster2;
  int soft;

  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), false);

  cluster.add (db::PolygonRef (poly, repo), 0);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), false);

  cluster2.add (db::PolygonRef (poly, repo), 1);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), true);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (db::Trans (db::Vector (10, 20))), conn, soft), true);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (db::Trans (db::Vector (0, 1000))), conn, soft), true);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (db::Trans (db::Vector (0, 1001))), conn, soft), false);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (db::Trans (db::Vector (0, 2000))), conn, soft), false);

  cluster.clear ();
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), false);
  cluster.add (db::PolygonRef (poly, repo), 2);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), false); //  not connected

  cluster.clear ();
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), false);
  cluster.add (db::PolygonRef (poly, repo), 1);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conn, soft), true);
}

TEST(11_LocalClusterInteractNetShapeScanner)
{
  db::GenericRepository repo;
  db::Polygon poly (db::Box (0, 0, 10, 10));

  db::local_cluster<db::NetShape> cluster;
  db::local_cluster<db::NetShape> cluster2;
  for (db::Coord x = 0; x < 6000; x += 100) {
    db::PolygonRef ref (poly, repo);
    ref.transform (db::Disp (db::Vector (x, 0)));
    cluster.add (db::NetShape (ref), 0);
    cluster2.add (db::NetShape (ref), 1);
  }

  db::Connectivity conn;
  conn.connect (0, 1);

  int soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (db::Trans (db::Vector (10, 0))), conn, soft), true);
  EXPECT_EQ (soft, 0);

  soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (db::Trans (db::Vector (20, 0))), conn, soft), false);
}

TEST(12_LocalClusterInteractLayerPruning)
{
  db::GenericRepository repo;

  db::Polygon poly;
  tl::from_string ("(0,0;0,1000;1000,1000;1000,0)", poly);

  db::local_cluster<db::PolygonRef> cluster;
  db::local_cluster<db::PolygonRef> cluster2;

  //  Layers 1/2 and 4/5 are overlapping spatial clutter.  Only layers 0/3
  //  participate in the first connectivity graph.
  cluster.add (db::PolygonRef (poly, repo), 0);
  cluster.add (db::PolygonRef (poly, repo), 1);
  cluster.add (db::PolygonRef (poly, repo), 2);
  cluster2.add (db::PolygonRef (poly, repo), 3);
  cluster2.add (db::PolygonRef (poly, repo), 4);
  cluster2.add (db::PolygonRef (poly, repo), 5);

  int soft = std::numeric_limits<int>::max ();

  db::Connectivity hard;
  hard.connect (0, 3);
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), hard, soft), true);
  EXPECT_EQ (soft, 0);

  db::Connectivity one_soft;
  one_soft.soft_connect (0, 3);
  soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), one_soft, soft), true);
  EXPECT_EQ (soft, -1);

  //  Opposite soft directions collapse to a hard interaction regardless of
  //  callback order.
  db::Connectivity conflicting_soft;
  conflicting_soft.soft_connect (0, 3);
  conflicting_soft.soft_connect (4, 1);
  soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), conflicting_soft, soft), true);
  EXPECT_EQ (soft, 0);

  //  A hard interaction likewise dominates a soft one.
  db::Connectivity hard_and_soft;
  hard_and_soft.connect (0, 3);
  hard_and_soft.soft_connect (4, 1);
  soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), hard_and_soft, soft), true);
  EXPECT_EQ (soft, 0);

  //  Shape collection intentionally follows the unpruned legacy path.  It
  //  must report only the connected target shape, in its original layer.
  std::map<unsigned int, std::vector<const db::PolygonRef *> > interacting;
  soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (cluster2, db::ICplxTrans (), hard, soft, 0, &interacting), true);
  EXPECT_EQ (soft, 0);
  EXPECT_EQ (interacting.size (), size_t (1));
  EXPECT_EQ (interacting [3].size (), size_t (1));
}

TEST(13_LocalClusterInteractLayerBboxCache)
{
  db::GenericRepository repo;

  db::local_cluster<db::PolygonRef> cluster;
  db::local_cluster<db::PolygonRef> other;

  //  Layer 0's aggregate box spans the common region, but neither of its
  //  actual shapes does.  Layer 2 contains the only exact interaction.
  cluster.add (db::PolygonRef (db::Polygon (db::Box (0, 0, 10, 10)), repo), 0);
  cluster.add (db::PolygonRef (db::Polygon (db::Box (90, 0, 100, 10)), repo), 0);
  cluster.add (db::PolygonRef (db::Polygon (db::Box (45, 0, 55, 10)), repo), 2);
  other.add (db::PolygonRef (db::Polygon (db::Box (45, 0, 55, 10)), repo), 1);

  db::Connectivity gap_connection;
  gap_connection.connect (0, 1);

  int soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (other, db::ICplxTrans (), gap_connection, soft), false);
  EXPECT_EQ (soft, std::numeric_limits<int>::max ());

  std::map<unsigned int, std::vector<const db::PolygonRef *> > interacting_this, interacting_other;
  EXPECT_EQ (cluster.interacts (other, db::ICplxTrans (), gap_connection, soft,
                                &interacting_this, &interacting_other), false);
  EXPECT_EQ (interacting_this.empty (), true);
  EXPECT_EQ (interacting_other.empty (), true);

  //  The false-positive hard pair must not override the only real, soft pair.
  db::Connectivity mixed_connection;
  mixed_connection.connect (0, 1);
  mixed_connection.soft_connect (2, 1);
  soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (other, db::ICplxTrans (), mixed_connection, soft), true);
  EXPECT_EQ (soft, -1);

  soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (other, db::ICplxTrans (), mixed_connection, soft,
                                &interacting_this, &interacting_other), true);
  EXPECT_EQ (soft, -1);
  EXPECT_EQ (interacting_this.size (), size_t (1));
  EXPECT_EQ (interacting_this [2].size (), size_t (1));
  EXPECT_EQ (interacting_other.size (), size_t (1));
  EXPECT_EQ (interacting_other [1].size (), size_t (1));

  //  Warm the caches, then verify every mutating and copying path rebuilds or
  //  preserves the per-layer boxes along with the cluster box.
  db::local_cluster<db::PolygonRef> changing;
  changing.add (db::PolygonRef (db::Polygon (db::Box (0, 0, 10, 10)), repo), 0);
  EXPECT_EQ (changing.interacts (other, db::ICplxTrans (), gap_connection, soft), false);
  changing.add (db::PolygonRef (db::Polygon (db::Box (45, 0, 55, 10)), repo), 0);
  EXPECT_EQ (changing.interacts (other, db::ICplxTrans (), gap_connection, soft), true);

  db::local_cluster<db::PolygonRef> sorted_copy (changing);
  EXPECT_EQ (sorted_copy.interacts (other, db::ICplxTrans (), gap_connection, soft), true);

  changing.clear ();
  changing.add (db::PolygonRef (db::Polygon (db::Box (0, 0, 10, 10)), repo), 0);
  EXPECT_EQ (changing.interacts (other, db::ICplxTrans (), gap_connection, soft), false);

  db::local_cluster<db::PolygonRef> donor;
  donor.add (db::PolygonRef (db::Polygon (db::Box (45, 0, 55, 10)), repo), 0);
  EXPECT_EQ (donor.bbox ().to_string (), "(45,0;55,10)");
  changing.join_with (donor);
  EXPECT_EQ (changing.interacts (other, db::ICplxTrans (), gap_connection, soft), true);

  changing.clear ();
  changing.add (db::PolygonRef (db::Polygon (db::Box (0, 0, 10, 10)), repo), 0);
  EXPECT_EQ (changing.bbox ().to_string (), "(0,0;10,10)");
  changing.add (db::PolygonRef (db::Polygon (db::Box (45, 0, 55, 10)), repo), 0);
  db::local_cluster<db::PolygonRef> dirty_copy (changing);
  EXPECT_EQ (dirty_copy.interacts (other, db::ICplxTrans (), gap_connection, soft), true);
}

TEST(14_LocalClusterInteractSparseCandidateLayers)
{
  db::GenericRepository repo;

  const unsigned int source_layer = std::numeric_limits<unsigned int>::max () - 1;
  const unsigned int exact_layer = std::numeric_limits<unsigned int>::max ();

  db::local_cluster<db::PolygonRef> cluster;
  db::local_cluster<db::PolygonRef> other;

  cluster.add (db::PolygonRef (db::Polygon (db::Box (45, 0, 55, 10)), repo), source_layer);

  //  This connected layer's aggregate box touches the common region, but its
  //  individual shapes do not.  A boolean scan must continue to the later
  //  connected layer containing the only exact interaction.
  other.add (db::PolygonRef (db::Polygon (db::Box (0, 0, 10, 10)), repo), 3);
  other.add (db::PolygonRef (db::Polygon (db::Box (90, 0, 100, 10)), repo), 3);

  //  An exact but disconnected layer exercises gaps in the sorted candidate
  //  vectors without relying on dense layer IDs.
  other.add (db::PolygonRef (db::Polygon (db::Box (45, 0, 55, 10)), repo), 4);
  other.add (db::PolygonRef (db::Polygon (db::Box (45, 0, 55, 10)), repo), exact_layer);

  db::Connectivity conn;
  conn.connect (source_layer, 1);  //  Connected, but absent from other.
  conn.connect (source_layer, 3);
  conn.soft_connect (source_layer, exact_layer);

  int soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (other, db::ICplxTrans (), conn, soft), true);
  EXPECT_EQ (soft, -1);

  //  Reporting intentionally retains the all-layer scanner path.  It must
  //  still reject the disconnected shape and collect the sparse-ID pair.
  std::map<unsigned int, std::vector<const db::PolygonRef *> > interacting_this, interacting_other;
  soft = std::numeric_limits<int>::max ();
  EXPECT_EQ (cluster.interacts (other, db::ICplxTrans (), conn, soft,
                                &interacting_this, &interacting_other), true);
  EXPECT_EQ (soft, -1);
  EXPECT_EQ (interacting_this.size (), size_t (1));
  EXPECT_EQ (interacting_this [source_layer].size (), size_t (1));
  EXPECT_EQ (interacting_other.size (), size_t (1));
  EXPECT_EQ (interacting_other [exact_layer].size (), size_t (1));
}

static std::string obj2string (const db::PolygonRef &ref)
{
  return ref.obj ().transformed (ref.trans ()).to_string ();
}

static std::string obj2string (const db::Edge &ref)
{
  return ref.to_string ();
}

template <class T>
static std::string local_cluster_to_string (const db::local_cluster<T> &cluster, const db::Connectivity &conn)
{
  std::string res;
  for (db::Connectivity::all_layer_iterator l = conn.begin_layers (); l != conn.end_layers (); ++l) {
    for (typename db::local_cluster<T>::shape_iterator s = cluster.begin (*l); ! s.at_end (); ++s) {
      if (! res.empty ()) {
        res += ";";
      }
      res += "[" + tl::to_string (*l) + "]" + obj2string (*s);
    }
  }
  for (typename db::local_cluster<T>::attr_iterator a = cluster.begin_attr (); a != cluster.end_attr (); ++a) {
    res += "%" + tl::to_string (*a);
  }
  for (typename db::local_cluster<T>::global_nets_iterator g = cluster.begin_global_nets (); g != cluster.end_global_nets (); ++g) {
    res += "+" + conn.global_net_name (*g);
  }
  return res;
}

template <class T>
static std::string local_clusters_to_string (const db::local_clusters<T> &clusters, const db::Connectivity &conn)
{
  std::string s;
  for (typename db::local_clusters<T>::const_iterator c = clusters.begin (); c != clusters.end (); ++c) {
    if (! s.empty ()) {
      s += "\n";
    }
    s += "#" + tl::to_string (c->id ()) + ":" + local_cluster_to_string (*c, conn);
  }
  for (typename db::local_clusters<T>::const_iterator c = clusters.begin (); c != clusters.end (); ++c) {
    auto sc = clusters.upward_soft_connections (c->id ());
    for (auto i = sc.begin (); i != sc.end (); ++i) {
      if (! s.empty ()) {
        s += "\n";
      }
      s += "(#" + tl::to_string (*i) + "->#" + tl::to_string (c->id ()) + ")";
    }
  }
  return s;
}

TEST(12_LocalClusterSplitByAreaRatio)
{
  db::GenericRepository repo;
  db::Connectivity conn;
  conn.connect (0);
  conn.connect (1);
  conn.connect (2);

  db::local_cluster<db::PolygonRef> cluster (17);
  cluster.add (db::PolygonRef (db::Polygon (db::Box (0, 0, 20, 20)), repo), 0);
  cluster.add (db::PolygonRef (db::Polygon (db::Box (0, 0, 1000, 20)), repo), 0);
  cluster.add (db::PolygonRef (db::Polygon (db::Box (1000, 0, 1020, 1000)), repo), 1);
  cluster.add (db::PolygonRef (db::Polygon (db::Box (0, 1000, 1000, 1020)), repo), 2);

  std::list<db::local_cluster<db::PolygonRef> > out;
  std::back_insert_iterator<std::list<db::local_cluster<db::PolygonRef> > > iout = std::back_inserter (out);
  size_t n = cluster.split (10.0, iout);

  EXPECT_EQ (n, size_t (3));
  EXPECT_EQ (out.size (), size_t (3));

  std::list<db::local_cluster<db::PolygonRef> >::const_iterator i = out.begin ();
  EXPECT_EQ (local_cluster_to_string (*i, conn), "[0](0,0;0,20;20,20;20,0);[0](0,0;0,20;1000,20;1000,0)");
  EXPECT_EQ (i->id (), size_t (17));
  ++i;
  EXPECT_EQ (local_cluster_to_string (*i, conn), "[1](1000,0;1000,1000;1020,1000;1020,0)");
  EXPECT_EQ (i->id (), size_t (17));
  ++i;
  EXPECT_EQ (local_cluster_to_string (*i, conn), "[2](0,1000;0,1020;1000,1020;1000,1000)");
  EXPECT_EQ (i->id (), size_t (17));
}

TEST(20_LocalClustersBasic)
{
  db::Layout layout;
  db::Cell &cell = layout.cell (layout.add_cell ("TOP"));
  db::GenericRepository &repo = layout.shape_repository ();

  db::Connectivity conn;
  conn.connect (0);
  conn.connect (1);
  conn.connect (2);
  conn.connect (0, 1);
  conn.connect (0, 2);

  db::Polygon poly;
  tl::from_string ("(0,0;0,1000;1000,1000;1000,0)", poly);

  cell.shapes (0).insert (db::PolygonRef (poly, repo));

  db::local_clusters<db::PolygonRef> clusters;
  EXPECT_EQ (local_clusters_to_string (clusters, conn), "");

  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn), "#1:[0](0,0;0,1000;1000,1000;1000,0)");

  //  one more shape
  cell.shapes (0).insert (db::PolygonRef (poly.transformed (db::Trans (db::Vector (10, 20))), repo));

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn), "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20)");

  //  one more shape creating a new cluster
  cell.shapes (2).insert (db::PolygonRef (poly.transformed (db::Trans (db::Vector (0, 1100))), repo));

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn),
    "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20)\n"
    "#2:[2](0,1100;0,2100;1000,2100;1000,1100)"
  );

  //  one more shape connecting these
  cell.shapes (2).insert (db::PolygonRef (poly.transformed (db::Trans (db::Vector (0, 1000))), repo));

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn),
    "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20);[2](0,1000;0,2000;1000,2000;1000,1000);[2](0,1100;0,2100;1000,2100;1000,1100)"
  );

  //  one more shape opening a new cluster
  cell.shapes (1).insert (db::PolygonRef (poly.transformed (db::Trans (db::Vector (0, 1100))), repo));

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn),
    "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20);[2](0,1000;0,2000;1000,2000;1000,1000);[2](0,1100;0,2100;1000,2100;1000,1100)\n"
    "#2:[1](0,1100;0,2100;1000,2100;1000,1100)"
  );
}

TEST(20_LocalClustersNetShapeScanner)
{
  db::Layout layout;
  unsigned int layer = layout.insert_layer (db::LayerProperties (1, 0));
  db::Cell &cell = layout.cell (layout.add_cell ("TOP"));

  for (db::Coord x = 0; x < 12000; x += 100) {
    db::Polygon poly (db::Box (x, 0, x + 10, 10));
    cell.shapes (layer).insert (db::PolygonRef (poly, layout.shape_repository ()));
  }

  db::Connectivity conn;
  conn.connect (layer);

  db::local_clusters<db::NetShape> clusters;
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (clusters.size (), size_t (120));
  EXPECT_EQ (clusters.bbox ().to_string (), "(0,0;11910,10)");
}

TEST(21_LocalClustersBasicWithAttributes)
{
  db::Layout layout;
  db::Cell &cell = layout.cell (layout.add_cell ("TOP"));
  db::GenericRepository &repo = layout.shape_repository ();

  db::Connectivity conn;
  conn.connect (0);
  conn.connect (1);
  conn.connect (2);
  conn.connect (0, 1);
  conn.connect (0, 2);

  db::Polygon poly;
  tl::from_string ("(0,0;0,1000;1000,1000;1000,0)", poly);

  cell.shapes (0).insert (db::PolygonRef (poly, repo));

  db::local_clusters<db::PolygonRef> clusters;
  EXPECT_EQ (local_clusters_to_string (clusters, conn), "");

  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn), "#1:[0](0,0;0,1000;1000,1000;1000,0)");

  //  one more shape
  cell.shapes (0).insert (db::PolygonRefWithProperties (db::PolygonRef (poly.transformed (db::Trans (db::Vector (10, 20))), repo), 1));

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn), "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20)%1");

  //  one more shape creating a new cluster
  cell.shapes (2).insert (db::PolygonRefWithProperties (db::PolygonRef (poly.transformed (db::Trans (db::Vector (0, 1100))), repo), 2));

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn),
    "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20)%1\n"
    "#2:[2](0,1100;0,2100;1000,2100;1000,1100)%2"
  );

  //  one more shape connecting these
  cell.shapes (2).insert (db::PolygonRefWithProperties (db::PolygonRef (poly.transformed (db::Trans (db::Vector (0, 1000))), repo), 3));

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn),
    "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20);[2](0,1000;0,2000;1000,2000;1000,1000);[2](0,1100;0,2100;1000,2100;1000,1100)%1%2%3"
  );

  //  one more shape opening a new cluster
  cell.shapes (1).insert (db::PolygonRefWithProperties (db::PolygonRef (poly.transformed (db::Trans (db::Vector (0, 1100))), repo), 4));

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn),
    "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20);[2](0,1000;0,2000;1000,2000;1000,1000);[2](0,1100;0,2100;1000,2100;1000,1100)%1%2%3\n"
    "#2:[1](0,1100;0,2100;1000,2100;1000,1100)%4"
  );
}

TEST(22_LocalClustersWithGlobal)
{
  db::Layout layout;
  db::Cell &cell = layout.cell (layout.add_cell ("TOP"));
  db::GenericRepository &repo = layout.shape_repository ();

  db::Connectivity conn;
  conn.connect (0);
  conn.connect (1);
  conn.connect (2);
  conn.connect (0, 1);
  conn.connect (0, 2);

  db::Polygon poly;
  tl::from_string ("(0,0;0,1000;1000,1000;1000,0)", poly);

  cell.shapes (0).insert (db::PolygonRef (poly, repo));

  db::local_clusters<db::PolygonRef> clusters;
  EXPECT_EQ (local_clusters_to_string (clusters, conn), "");

  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn), "#1:[0](0,0;0,1000;1000,1000;1000,0)");

  //  one more shape
  cell.shapes (0).insert (db::PolygonRefWithProperties (db::PolygonRef (poly.transformed (db::Trans (db::Vector (10, 20))), repo), 1));

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn), "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20)%1");

  //  one more shape creating a new cluster
  cell.shapes (2).insert (db::PolygonRefWithProperties (db::PolygonRef (poly.transformed (db::Trans (db::Vector (0, 1100))), repo), 2));

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn),
    "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20)%1\n"
    "#2:[2](0,1100;0,2100;1000,2100;1000,1100)%2"
  );

  conn.connect_global (0, "GLOBAL");

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn),
    "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20)%1+GLOBAL\n"
    "#2:[2](0,1100;0,2100;1000,2100;1000,1100)%2"
  );

  conn.connect_global (2, "GLOBAL2");

  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn),
    "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20)%1+GLOBAL\n"
    "#2:[2](0,1100;0,2100;1000,2100;1000,1100)%2+GLOBAL2"
  );

  conn.connect_global (0, "GLOBAL2");

  //  now, GLOBAL2 will connect these clusters
  clusters.clear ();
  clusters.build_clusters (cell, conn);
  EXPECT_EQ (local_clusters_to_string (clusters, conn),
    "#1:[0](0,0;0,1000;1000,1000;1000,0);[0](10,20;10,1020;1010,1020;1010,20);[2](0,1100;0,2100;1000,2100;1000,1100)%1%2+GLOBAL+GLOBAL2"
  );
}

TEST(23_LocalClustersWithEdges)
{
  db::Layout layout;
  db::Cell &cell = layout.cell (layout.add_cell ("TOP"));

  db::Edge edge;

  tl::from_string ("(0,0;0,500)", edge);
  cell.shapes (0).insert (edge);

  tl::from_string ("(0,500;0,1000)", edge);
  cell.shapes (0).insert (edge);

  tl::from_string ("(0,1000;2000,1000)", edge);
  cell.shapes (0).insert (edge);

  tl::from_string ("(2000,1000;2000,500)", edge);
  cell.shapes (0).insert (edge);

  tl::from_string ("(2000,500;1000,250)", edge);
  cell.shapes (0).insert (edge);

  tl::from_string ("(1500,375;0,0)", edge);
  cell.shapes (0).insert (edge);

  {
    //  edge clusters are for intra-layer mainly
    db::Connectivity conn;
    conn.connect (0);

    db::local_clusters<db::Edge> clusters;
    clusters.build_clusters (cell, conn);
    EXPECT_EQ (local_clusters_to_string (clusters, conn),
      "#1:[0](0,0;0,500);[0](0,500;0,1000)\n"
      "#2:[0](2000,500;1000,250);[0](1500,375;0,0)\n"
      "#3:[0](0,1000;2000,1000)\n"
      "#4:[0](2000,1000;2000,500)"
    );
  }

  {
    //  edge clusters are for intra-layer mainly
    db::Connectivity conn (db::Connectivity::EdgesConnectByPoints);
    conn.connect (0);

    db::local_clusters<db::Edge> clusters;
    clusters.build_clusters (cell, conn);
    EXPECT_EQ (local_clusters_to_string (clusters, conn), "#1:[0](0,0;0,500);[0](0,500;0,1000);[0](1500,375;0,0);[0](0,1000;2000,1000);[0](2000,1000;2000,500);[0](2000,500;1000,250)");
  }
}

TEST(24_LocalClustersWithSoftConnections)
{
  db::Layout layout;
  db::Cell &cell = layout.cell (layout.add_cell ("TOP"));
  db::GenericRepository &repo = layout.shape_repository ();

  auto dbu = db::CplxTrans (layout.dbu ()).inverted ();

  unsigned int nwell = 0;
  unsigned int ntie = 1;
  unsigned int ptie = 2;
  unsigned int contact = 3;
  unsigned int metal1 = 4;

  cell.shapes (nwell).insert (db::PolygonRef (dbu * db::DPolygon (db::DBox (0.0, 4.0, 2.0, 8.0)), repo));
  cell.shapes (ntie).insert (db::PolygonRef (dbu * db::DPolygon (db::DBox (0.5, 5.0, 1.5, 7.0)), repo));
  cell.shapes (contact).insert (db::PolygonRef (dbu * db::DPolygon (db::DBox (0.8, 6.0, 1.2, 6.5)), repo));
  cell.shapes (metal1).insert (db::PolygonRef (dbu * db::DPolygon (db::DBox (0.0, 5.0, 2.0, 7.0)), repo));

  cell.shapes (ptie).insert (db::PolygonRef (dbu * db::DPolygon (db::DBox (0.5, 1.0, 1.5, 3.0)), repo));
  cell.shapes (contact).insert (db::PolygonRef (dbu * db::DPolygon (db::DBox (0.8, 2.0, 1.2, 2.5)), repo));
  cell.shapes (metal1).insert (db::PolygonRef (dbu * db::DPolygon (db::DBox (0.0, 1.0, 2.0, 3.0)), repo));

  db::Connectivity conn;
  conn.connect (nwell);
  conn.connect (ntie);
  conn.connect (ptie);
  conn.connect (contact);
  conn.connect (metal1);
  conn.soft_connect (ntie, nwell);
  conn.soft_connect (contact, ntie);
  conn.connect (metal1, contact);

  {
    db::local_clusters<db::PolygonRef> clusters;
    clusters.build_clusters (cell, conn);
    EXPECT_EQ (local_clusters_to_string (clusters, conn),
      "#1:[0](0,4000;0,8000;2000,8000;2000,4000)\n"
      "#2:[1](500,5000;500,7000;1500,7000;1500,5000)\n"
      "#3:[3](800,6000;800,6500;1200,6500;1200,6000);[4](0,5000;0,7000;2000,7000;2000,5000)\n"
      "#4:[3](800,2000;800,2500;1200,2500;1200,2000);[4](0,1000;0,3000;2000,3000;2000,1000)\n"
      "#5:[2](500,1000;500,3000;1500,3000;1500,1000)\n"
      "(#2->#1)\n"
      "(#3->#2)"
    );
  }

  conn.soft_connect (contact, ptie);

  {
    db::local_clusters<db::PolygonRef> clusters;
    clusters.build_clusters (cell, conn);
    EXPECT_EQ (local_clusters_to_string (clusters, conn),
      "#1:[0](0,4000;0,8000;2000,8000;2000,4000)\n"
      "#2:[1](500,5000;500,7000;1500,7000;1500,5000)\n"
      "#3:[3](800,6000;800,6500;1200,6500;1200,6000);[4](0,5000;0,7000;2000,7000;2000,5000)\n"
      "#4:[2](500,1000;500,3000;1500,3000;1500,1000)\n"
      "#5:[3](800,2000;800,2500;1200,2500;1200,2000);[4](0,1000;0,3000;2000,3000;2000,1000)\n"
      "(#2->#1)\n"
      "(#3->#2)\n"
      "(#5->#4)"
    );
  }

  conn.soft_connect_global (ptie, "BULK");

  {
    db::local_clusters<db::PolygonRef> clusters;
    clusters.build_clusters (cell, conn);
    EXPECT_EQ (local_clusters_to_string (clusters, conn),
      "#1:[0](0,4000;0,8000;2000,8000;2000,4000)\n"
      "#2:[1](500,5000;500,7000;1500,7000;1500,5000)\n"
      "#3:[3](800,6000;800,6500;1200,6500;1200,6000);[4](0,5000;0,7000;2000,7000;2000,5000)\n"
      "#4:[2](500,1000;500,3000;1500,3000;1500,1000)\n"
      "#5:[3](800,2000;800,2500;1200,2500;1200,2000);[4](0,1000;0,3000;2000,3000;2000,1000)\n"
      "#6:+BULK\n"
      "(#2->#1)\n"
      "(#3->#2)\n"
      "(#5->#4)\n"
      "(#4->#6)"
    );
  }
}

TEST(30_LocalConnectedClusters)
{
  db::Layout layout;
  db::cell_index_type ci1 = layout.add_cell ("C1");
  db::cell_index_type ci2 = layout.add_cell ("C2");
  db::cell_index_type ci3 = layout.add_cell ("C3");

  db::Instance i1 = layout.cell (ci1).insert (db::CellInstArray (db::CellInst (ci2), db::Trans ()));
  db::Instance i2 = layout.cell (ci2).insert (db::CellInstArray (db::CellInst (ci3), db::Trans ()));

  db::connected_clusters<db::PolygonRef> cc;

  db::connected_clusters<db::PolygonRef>::connections_type x;
  db::connected_clusters<db::PolygonRef>::connections_type::const_iterator ix;

  x = cc.connections_for_cluster (1);
  EXPECT_EQ (x.size (), size_t (0));
  x = cc.connections_for_cluster (2);
  EXPECT_EQ (x.size (), size_t (0));

  //  after this:
  //   [#1] -> i1:#1
  //        -> i2:#2
  cc.add_connection (1, db::ClusterInstance (1, db::InstElement (i1)));
  cc.add_connection (1, db::ClusterInstance (2, db::InstElement (i2)));

  x = cc.connections_for_cluster (1);
  EXPECT_EQ (x.size (), size_t (2));
  x = cc.connections_for_cluster (2);
  EXPECT_EQ (x.size (), size_t (0));

  //  after this:
  //   [#1] -> i1:#1
  //        -> i2:#2
  //   [#2] -> i2:#1
  cc.add_connection (2, db::ClusterInstance (1, db::InstElement (i2)));
  x = cc.connections_for_cluster (2);
  EXPECT_EQ (x.size (), size_t (1));

  cc.join_cluster_with (1, 2);
  x = cc.connections_for_cluster (1);
  EXPECT_EQ (x.size (), size_t (3));
  ix = x.begin ();
  EXPECT_EQ (ix->id (), size_t (1));
  EXPECT_EQ (*ix == db::ClusterInstance (ix->id (), i1.cell_index (), i1.complex_trans (), i1.prop_id ()), true);
  ++ix;
  EXPECT_EQ (ix->id (), size_t (2));
  EXPECT_EQ (*ix == db::ClusterInstance (ix->id (), i2.cell_index (), i2.complex_trans (), i2.prop_id ()), true);
  ++ix;
  EXPECT_EQ (ix->id (), size_t (1));
  EXPECT_EQ (*ix == db::ClusterInstance (ix->id (), i2.cell_index (), i2.complex_trans (), i2.prop_id ()), true);

  x = cc.connections_for_cluster (2);
  EXPECT_EQ (x.size (), size_t (0));

  //  after this:
  //   [#1] -> i1:#1
  //        -> i2:#2
  //   [#2] -> i2:#1
  //        -> i1:#3
  cc.add_connection (2, db::ClusterInstance (3, db::InstElement (i1)));

  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (3, db::InstElement (i1))), size_t (2));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (2, db::InstElement (i1))), size_t (0));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (2, db::InstElement (i2))), size_t (1));

  //  after this:
  //   [#1] -> i1:#1
  //        -> i2:#2
  //        -> i2:#1
  //        -> i1:#3
  cc.join_cluster_with (1, 2);
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (3, db::InstElement (i1))), size_t (1));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (1, db::InstElement (i1))), size_t (1));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (2, db::InstElement (i1))), size_t (0));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (2, db::InstElement (i2))), size_t (1));

  x = cc.connections_for_cluster (1);
  EXPECT_EQ (x.size (), size_t (4));
  x = cc.connections_for_cluster (2);
  EXPECT_EQ (x.size (), size_t (0));
}

TEST(31_ConnectedClustersReverseLookupHash)
{
  typedef db::connected_clusters<db::PolygonRef> clusters_type;

  clusters_type cc;

  db::ICplxTrans t (1.0, 90.0, false, db::Vector (100, 200));
  db::ICplxTrans t_equivalent (db::Trans (1, false, db::Vector (100, 200)));
  db::ClusterInstance base (7, 11, t, 13);
  db::ClusterInstance equivalent (7, 11, t_equivalent, 13);

  EXPECT_EQ (base == equivalent, true);
  cc.add_connection (101, base);
  EXPECT_EQ (cc.find_cluster_with_connection (base), size_t (101));
  EXPECT_EQ (cc.find_cluster_with_connection (equivalent), size_t (101));

  //  Fuzzy-equal transformations must remain interchangeable as hash keys,
  //  even when magnification straddles a would-be epsilon-sized hash bin.
  db::ICplxTrans fuzzy_t (1.0, 30.0, false, db::Vector (500, 600));
  db::ICplxTrans fuzzy_t_equivalent (1.0 + 0.75 * db::epsilon, 30.0, false, db::Vector (500, 600));
  db::ClusterInstance fuzzy (27, 29, fuzzy_t, 31);
  db::ClusterInstance fuzzy_equivalent (27, 29, fuzzy_t_equivalent, 31);
  EXPECT_EQ (fuzzy == fuzzy_equivalent, true);
  cc.add_connection (102, fuzzy);
  EXPECT_EQ (cc.find_cluster_with_connection (fuzzy_equivalent), size_t (102));

  //  ICplxTrans retains its displacement as doubles even when its public
  //  displacement accessor rounds to integer coordinates.  Exercise adjacent
  //  rounded buckets on each axis, both signs and both lookup directions.
  const double displacement_epsilon = db::coord_traits<double>::prec ();
  const double boundary_delta = 0.4 * displacement_epsilon;
  auto check_boundary = [&] (size_t serial, const db::DVector &da, const db::DVector &db_disp,
                             const db::Vector &rounded_a, const db::Vector &rounded_b)
  {
    db::ICplxTrans ta (da, 0.0, 1.0, 1.0);
    db::ICplxTrans tb (db_disp, 0.0, 1.0, 1.0);

    EXPECT_EQ (ta.disp ().x (), rounded_a.x ());
    EXPECT_EQ (ta.disp ().y (), rounded_a.y ());
    EXPECT_EQ (tb.disp ().x (), rounded_b.x ());
    EXPECT_EQ (tb.disp ().y (), rounded_b.y ());

    for (size_t reverse = 0; reverse < 2; ++reverse) {
      const size_t child_id = 20000 + serial * 2 + reverse;
      const size_t parent_id = 10000 + serial * 2 + reverse;
      db::ClusterInstance a (child_id, 41, ta, 43);
      db::ClusterInstance b (child_id, 41, tb, 43);
      EXPECT_EQ (a == b, true);
      if (reverse == 0) {
        cc.add_connection (parent_id, a);
        EXPECT_EQ (cc.find_cluster_with_connection (b), parent_id);
      } else {
        cc.add_connection (parent_id, b);
        EXPECT_EQ (cc.find_cluster_with_connection (a), parent_id);
      }
    }
  };

  check_boundary (0,
                  db::DVector (0.5 - boundary_delta, 10.0), db::DVector (0.5 + boundary_delta, 10.0),
                  db::Vector (0, 10), db::Vector (1, 10));
  check_boundary (1,
                  db::DVector (-0.5 - boundary_delta, 10.0), db::DVector (-0.5 + boundary_delta, 10.0),
                  db::Vector (-1, 10), db::Vector (0, 10));
  check_boundary (2,
                  db::DVector (10.0, 0.5 - boundary_delta), db::DVector (10.0, 0.5 + boundary_delta),
                  db::Vector (10, 0), db::Vector (10, 1));
  check_boundary (3,
                  db::DVector (10.0, -0.5 - boundary_delta), db::DVector (10.0, -0.5 + boundary_delta),
                  db::Vector (10, -1), db::Vector (10, 0));
  check_boundary (4,
                  db::DVector (0.5 - boundary_delta, 0.5 - boundary_delta),
                  db::DVector (0.5 + boundary_delta, 0.5 + boundary_delta),
                  db::Vector (0, 0), db::Vector (1, 1));
  check_boundary (5,
                  db::DVector (-0.5 - boundary_delta, -0.5 - boundary_delta),
                  db::DVector (-0.5 + boundary_delta, -0.5 + boundary_delta),
                  db::Vector (-1, -1), db::Vector (0, 0));
  check_boundary (6,
                  db::DVector (0.5 - boundary_delta, -0.5 + boundary_delta),
                  db::DVector (0.5 + boundary_delta, -0.5 - boundary_delta),
                  db::Vector (0, 0), db::Vector (1, -1));

  //  A displacement next to the coordinate endpoints still lies close enough
  //  to a half-integer to request an adjacent bucket.  The absent bucket is
  //  outside Coord's range and must not be formed through signed overflow.
  //  This construction needs 32-bit coordinates: double cannot represent the
  //  half-integer next to a 64-bit Coord endpoint.
#if !HAVE_64BIT_COORD
  const db::Coord coord_max = std::numeric_limits<db::Coord>::max ();
  const db::Coord coord_min = std::numeric_limits<db::Coord>::min ();
  const double raw_max = double (coord_max) + 0.5 - boundary_delta;
  const double raw_min = double (coord_min) - 0.5 + boundary_delta;
  db::ICplxTrans endpoint_max_t (db::DVector (raw_max, raw_max), 0.0, 1.0, 1.0);
  db::ICplxTrans endpoint_min_t (db::DVector (raw_min, raw_min), 0.0, 1.0, 1.0);
  EXPECT_EQ (endpoint_max_t.disp (), db::Vector (coord_max, coord_max));
  EXPECT_EQ (endpoint_min_t.disp (), db::Vector (coord_min, coord_min));
  db::ClusterInstance endpoint_max (31000, 41, endpoint_max_t, 43);
  db::ClusterInstance endpoint_min (31001, 41, endpoint_min_t, 43);
  cc.add_connection (31002, endpoint_max);
  cc.add_connection (31003, endpoint_min);
  EXPECT_EQ (cc.find_cluster_with_connection (endpoint_max), size_t (31002));
  EXPECT_EQ (cc.find_cluster_with_connection (endpoint_min), size_t (31003));
#endif

  //  Adjacent rounded buckets alone are insufficient: the complete fuzzy
  //  ClusterInstance equality check must still reject displacements over the
  //  tolerance.
  const double non_equal_delta = 0.6 * displacement_epsilon;
  db::ICplxTrans negative_far_a (db::DVector (-0.5 - non_equal_delta, 20.0), 0.0, 1.0, 1.0);
  db::ICplxTrans negative_far_b (db::DVector (-0.5 + non_equal_delta, 20.0), 0.0, 1.0, 1.0);
  db::ClusterInstance negative_far_key_a (30000, 41, negative_far_a, 43);
  db::ClusterInstance negative_far_key_b (30000, 41, negative_far_b, 43);
  EXPECT_EQ (negative_far_key_a == negative_far_key_b, false);
  cc.add_connection (30001, negative_far_key_a);
  EXPECT_EQ (cc.find_cluster_with_connection (negative_far_key_b), size_t (0));

  //  Every component of ClusterInstance participates in the reverse lookup key.
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (8, 11, t, 13)), size_t (0));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (7, 12, t, 13)), size_t (0));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (7, 11, db::ICplxTrans (1.0, 90.0, false, db::Vector (101, 200)), 13)), size_t (0));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (7, 11, db::ICplxTrans (1.0, 0.0, false, db::Vector (100, 200)), 13)), size_t (0));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (7, 11, db::ICplxTrans (1.0, 90.0, true, db::Vector (100, 200)), 13)), size_t (0));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (7, 11, db::ICplxTrans (2.0, 90.0, false, db::Vector (100, 200)), 13)), size_t (0));
  EXPECT_EQ (cc.find_cluster_with_connection (db::ClusterInstance (7, 11, t, 14)), size_t (0));

  //  Renaming changes the child cluster ID in both lookup directions.
  cc.rename_connection (equivalent, 9);
  db::ClusterInstance renamed (9, 11, t, 13);
  EXPECT_EQ (cc.find_cluster_with_connection (base), size_t (0));
  EXPECT_EQ (cc.find_cluster_with_connection (renamed), size_t (101));
  EXPECT_EQ (cc.connections_for_cluster (101).front () == renamed, true);

  //  Joining parent clusters retargets every reverse entry from the joined cluster.
  db::ClusterInstance joined (17, 19, db::ICplxTrans (1.5, 45.0, true, db::Vector (-300, 400)), 23);
  cc.add_connection (202, joined);
  cc.join_cluster_with (101, 202);
  EXPECT_EQ (cc.find_cluster_with_connection (renamed), size_t (101));
  EXPECT_EQ (cc.find_cluster_with_connection (joined), size_t (101));
  EXPECT_EQ (cc.connections_for_cluster (202).empty (), true);

  //  Array members share cell, property and child-cluster IDs, but their member
  //  transformations must remain distinct keys.
  db::Layout layout;
  db::Cell &top = layout.cell (layout.add_cell ("TOP"));
  db::Cell &child = layout.cell (layout.add_cell ("CHILD"));
  db::Instance array = top.insert (db::CellInstArray (db::CellInst (child.cell_index ()), db::Trans (),
                                                     db::Vector (1000, 0), db::Vector (0, 2000), 2, 2));

  std::vector<db::ClusterInstance> array_members;
  for (db::CellInstArray::iterator ai = array.begin (); ! ai.at_end (); ++ai) {
    array_members.push_back (db::ClusterInstance (31, db::InstElement (array, ai)));
  }
  EXPECT_EQ (array_members.size (), size_t (4));

  for (size_t n = 0; n < array_members.size (); ++n) {
    cc.add_connection (300 + n, array_members [n]);
  }
  for (size_t n = 0; n < array_members.size (); ++n) {
    EXPECT_EQ (cc.find_cluster_with_connection (array_members [n]), size_t (300 + n));
  }
}

TEST(32_ConnectedClustersReverseLookupTableLifecycle)
{
  typedef db::connected_clusters<db::PolygonRef> clusters_type;

  //  These keys differ only in magnification.  Magnification is intentionally
  //  absent from the coarse hash because its equality is fuzzy, so this makes
  //  one long probe sequence while preserving distinct full keys.
  const size_t key_count = 320;
  const size_t old_child_id = 700;
  const size_t target_child_id = 701;
  std::vector<db::ClusterInstance> old_keys;
  old_keys.reserve (key_count);

  clusters_type cc;
  for (size_t n = 0; n < key_count; ++n) {
    old_keys.push_back (db::ClusterInstance (old_child_id, 77,
                                             db::ICplxTrans (1.0 + 0.01 * double (n), 0.0, false,
                                                             db::Vector (-123, 456)),
                                             99));
    cc.add_connection (1000 + n, old_keys.back ());
  }
  for (size_t n = 0; n < key_count; ++n) {
    EXPECT_EQ (cc.find_cluster_with_connection (old_keys [n]), size_t (1000 + n));
  }

  //  Precreate rename targets.  Renaming to an existing target erases the old
  //  reverse entry without immediately filling that slot, leaving alternating
  //  tombstones throughout the original probe sequence.
  std::vector<db::ClusterInstance> target_keys;
  target_keys.reserve ((key_count + 1) / 2);
  for (size_t n = 0; n < key_count; n += 2) {
    target_keys.push_back (db::ClusterInstance (target_child_id, old_keys [n]));
    cc.add_connection (2000 + n, target_keys.back ());
  }
  for (size_t n = 0; n < key_count; n += 2) {
    EXPECT_EQ (cc.find_cluster_with_connection (target_keys [n / 2]), size_t (2000 + n));
  }

  for (size_t n = 0; n < key_count; n += 2) {
    cc.rename_connection (old_keys [n], target_child_id);
  }
  for (size_t n = 0; n < key_count; ++n) {
    EXPECT_EQ (cc.find_cluster_with_connection (old_keys [n]), n % 2 == 0 ? size_t (0) : size_t (1000 + n));
  }
  for (size_t n = 0; n < key_count; n += 2) {
    EXPECT_EQ (cc.find_cluster_with_connection (target_keys [n / 2]), size_t (2000 + n));
  }

  //  Reuse some of the tombstones with fresh keys from the same coarse bucket.
  std::vector<db::ClusterInstance> replacement_keys;
  replacement_keys.reserve (64);
  for (size_t n = 0; n < 64; ++n) {
    replacement_keys.push_back (db::ClusterInstance (old_child_id, 77,
                                                     db::ICplxTrans (10.0 + 0.01 * double (n), 0.0, false,
                                                                     db::Vector (-123, 456)),
                                                     99));
    cc.add_connection (3000 + n, replacement_keys.back ());
  }

  auto verify_bulk = [&] (const clusters_type &clusters)
  {
    for (size_t n = 0; n < key_count; ++n) {
      EXPECT_EQ (clusters.find_cluster_with_connection (old_keys [n]), n % 2 == 0 ? size_t (0) : size_t (1000 + n));
    }
    for (size_t n = 0; n < key_count; n += 2) {
      EXPECT_EQ (clusters.find_cluster_with_connection (target_keys [n / 2]), size_t (2000 + n));
    }
    for (size_t n = 0; n < replacement_keys.size (); ++n) {
      EXPECT_EQ (clusters.find_cluster_with_connection (replacement_keys [n]), size_t (3000 + n));
    }
  };
  verify_bulk (cc);

  //  Adding the same fuzzy key updates the reverse mapping.  Joining the two
  //  parent clusters retargets it and removes the duplicate forward connection.
  clusters_type duplicates;
  db::ICplxTrans duplicate_t (1.0, 30.0, false, db::Vector (700, 800));
  db::ICplxTrans duplicate_t_equivalent (1.0 + 0.5 * db::epsilon, 30.0, false, db::Vector (700, 800));
  db::ClusterInstance duplicate_key (900, 901, duplicate_t, 902);
  db::ClusterInstance duplicate_key_equivalent (900, 901, duplicate_t_equivalent, 902);
  EXPECT_EQ (duplicate_key == duplicate_key_equivalent, true);
  duplicates.add_connection (41, duplicate_key);
  duplicates.add_connection (42, duplicate_key_equivalent);
  EXPECT_EQ (duplicates.find_cluster_with_connection (duplicate_key), size_t (42));
  duplicates.join_cluster_with (41, 42);
  EXPECT_EQ (duplicates.find_cluster_with_connection (duplicate_key_equivalent), size_t (41));
  EXPECT_EQ (duplicates.connections_for_cluster (41).size (), size_t (1));
  EXPECT_EQ (duplicates.connections_for_cluster (42).empty (), true);

  //  The table is copied by value in extraction code.  Verify independent
  //  copies, both move paths and valid reuse of moved-from objects.
  clusters_type copied (cc);
  clusters_type assigned;
  assigned = cc;
  verify_bulk (copied);
  verify_bulk (assigned);

  db::ClusterInstance source_only (910, 911, db::ICplxTrans (1.25, 15.0, false, db::Vector (12, 34)), 912);
  cc.add_connection (9000, source_only);
  EXPECT_EQ (cc.find_cluster_with_connection (source_only), size_t (9000));
  EXPECT_EQ (copied.find_cluster_with_connection (source_only), size_t (0));
  EXPECT_EQ (assigned.find_cluster_with_connection (source_only), size_t (0));

  clusters_type moved (std::move (copied));
  clusters_type move_assigned;
  move_assigned = std::move (assigned);
  verify_bulk (moved);
  verify_bulk (move_assigned);

  db::ClusterInstance reused_after_move_1 (920, 921, db::ICplxTrans (db::Vector (1, 2)), 922);
  db::ClusterInstance reused_after_move_2 (930, 931, db::ICplxTrans (db::Vector (3, 4)), 932);
  copied.add_connection (9001, reused_after_move_1);
  assigned.add_connection (9002, reused_after_move_2);
  EXPECT_EQ (copied.find_cluster_with_connection (reused_after_move_1), size_t (9001));
  EXPECT_EQ (assigned.find_cluster_with_connection (reused_after_move_2), size_t (9002));

  //  Keep only a few entries live while repeatedly renaming them.  This creates
  //  many deleted slots and drives same-capacity tombstone compaction.
  clusters_type churn;
  std::vector<db::ClusterInstance> live_keys;
  for (size_t n = 0; n < 8; ++n) {
    live_keys.push_back (db::ClusterInstance (50000 + n, 501,
                                             db::ICplxTrans (1.0 + 0.1 * double (n), 0.0, false,
                                                             db::Vector (-50, 60)),
                                             502));
    churn.add_connection (4000 + n, live_keys.back ());
  }
  for (size_t round = 0; round < 256; ++round) {
    for (size_t n = 0; n < live_keys.size (); ++n) {
      db::ClusterInstance old_key = live_keys [n];
      db::ClusterInstance new_key (50000 + (round + 1) * live_keys.size () + n, old_key);
      churn.rename_connection (old_key, new_key.id ());
      EXPECT_EQ (churn.find_cluster_with_connection (old_key), size_t (0));
      EXPECT_EQ (churn.find_cluster_with_connection (new_key), size_t (4000 + n));
      live_keys [n] = new_key;
    }
  }
}

static db::PolygonRef make_box (db::Layout &ly, const db::Box &box)
{
  return db::PolygonRef (db::Polygon (box), ly.shape_repository ());
}

TEST(40_HierClustersBasic)
{
  db::hier_clusters<db::PolygonRef> hc;

  db::Layout ly;
  unsigned int l1 = ly.insert_layer (db::LayerProperties (1, 0));

  db::Cell &top = ly.cell (ly.add_cell ("TOP"));
  top.shapes (l1).insert (make_box (ly, db::Box (0, 0, 1000, 1000)));

  db::Cell &c1 = ly.cell (ly.add_cell ("C1"));
  c1.shapes (l1).insert (make_box (ly, db::Box (0, 0, 2000, 500)));
  top.insert (db::CellInstArray (db::CellInst (c1.cell_index ()), db::Trans ()));

  db::Cell &c2 = ly.cell (ly.add_cell ("C2"));
  c2.shapes (l1).insert (make_box (ly, db::Box (0, 0, 500, 2000)));
  c2.insert (db::CellInstArray (db::CellInst (c1.cell_index ()), db::Trans ()));
  top.insert (db::CellInstArray (db::CellInst (c2.cell_index ()), db::Trans ()));

  db::Connectivity conn;
  conn.connect (l1, l1);

  hc.build (ly, top, conn);

  int n, nc;
  const db::connected_clusters<db::PolygonRef> *cluster;

  //  1 cluster in TOP with 2 connections
  n = 0;
  cluster = &hc.clusters_per_cell (top.cell_index ());
  for (db::connected_clusters<db::PolygonRef>::const_iterator i = cluster->begin (); i != cluster->end (); ++i) {
    ++n;
  }
  EXPECT_EQ (n, 1);
  EXPECT_EQ (cluster->bbox ().to_string (), "(0,0;1000,1000)")
  nc = 0;
  for (db::connected_clusters<db::PolygonRef>::connections_iterator i = cluster->begin_connections (); i != cluster->end_connections (); ++i) {
    nc += int (i->second.size ());
  }
  EXPECT_EQ (nc, 2);

  //  1 cluster in C1 without connection
  n = 0;
  cluster = &hc.clusters_per_cell (c1.cell_index ());
  for (db::connected_clusters<db::PolygonRef>::const_iterator i = cluster->begin (); i != cluster->end (); ++i) {
    ++n;
  }
  EXPECT_EQ (n, 1);
  EXPECT_EQ (cluster->bbox ().to_string (), "(0,0;2000,500)")
  nc = 0;
  for (db::connected_clusters<db::PolygonRef>::connections_iterator i = cluster->begin_connections (); i != cluster->end_connections (); ++i) {
    nc += int (i->second.size ());
  }
  EXPECT_EQ (nc, 0);

  //  1 cluster in C2 with one connection
  n = 0;
  cluster = &hc.clusters_per_cell (c2.cell_index ());
  for (db::connected_clusters<db::PolygonRef>::const_iterator i = cluster->begin (); i != cluster->end (); ++i) {
    ++n;
  }
  EXPECT_EQ (n, 1);
  EXPECT_EQ (cluster->bbox ().to_string (), "(0,0;500,2000)")
  nc = 0;
  for (db::connected_clusters<db::PolygonRef>::connections_iterator i = cluster->begin_connections (); i != cluster->end_connections (); ++i) {
    nc += int (i->second.size ());
  }
  EXPECT_EQ (nc, 1);
}

static std::string path2string (const db::Layout &ly, db::cell_index_type ci, const std::vector<db::ClusterInstance> &path)
{
  std::string res = ly.cell_name (ci);
  for (std::vector<db::ClusterInstance>::const_iterator p = path.begin (); p != path.end (); ++p) {
    res += "/";
    res += ly.cell_name (p->inst_cell_index ());
  }
  return res;
}

static std::string rcsiter2string (const db::Layout &ly, db::cell_index_type ci, db::recursive_cluster_shape_iterator<db::PolygonRef> si, db::cell_index_type ci2skip = std::numeric_limits<db::cell_index_type>::max ())
{
  std::string res;
  while (! si.at_end ()) {
    if (si.cell_index () == ci2skip) {
      si.skip_cell ();
      continue;
    }
    db::Polygon poly = si->obj ();
    poly.transform (si->trans ());
    poly.transform (si.trans ());
    if (! res.empty ()) {
      res += ";";
    }
    res += path2string (ly, ci, si.inst_path ());
    res += ":";
    res += poly.to_string ();
    ++si;
  }
  return res;
}

static std::string rciter2string (const db::Layout &ly, db::cell_index_type ci, db::recursive_cluster_iterator<db::PolygonRef> si)
{
  std::string res;
  while (! si.at_end ()) {
    if (! res.empty ()) {
      res += ";";
    }
    res += path2string (ly, ci, si.inst_path ());
    ++si;
  }
  return res;
}

TEST(41_HierClustersRecursiveClusterShapeIterator)
{
  db::hier_clusters<db::PolygonRef> hc;

  db::Layout ly;
  unsigned int l1 = ly.insert_layer (db::LayerProperties (1, 0));

  db::Cell &top = ly.cell (ly.add_cell ("TOP"));
  top.shapes (l1).insert (make_box (ly, db::Box (0, 0, 1000, 1000)));

  db::Cell &c1 = ly.cell (ly.add_cell ("C1"));
  c1.shapes (l1).insert (make_box (ly, db::Box (0, 0, 2000, 500)));
  top.insert (db::CellInstArray (db::CellInst (c1.cell_index ()), db::Trans (db::Vector (0, 10))));

  db::Cell &c2 = ly.cell (ly.add_cell ("C2"));
  c2.shapes (l1).insert (make_box (ly, db::Box (0, 0, 500, 2000)));
  c2.insert (db::CellInstArray (db::CellInst (c1.cell_index ()), db::Trans (db::Vector (0, 20))));
  top.insert (db::CellInstArray (db::CellInst (c2.cell_index ()), db::Trans (db::Vector (0, 30))));

  db::Connectivity conn;
  conn.connect (l1, l1);

  hc.build (ly, top, conn);

  std::string res;
  int n = 0;
  db::connected_clusters<db::PolygonRef> *cluster = &hc.clusters_per_cell (top.cell_index ());
  for (db::connected_clusters<db::PolygonRef>::const_iterator i = cluster->begin (); i != cluster->end (); ++i) {
    res = rcsiter2string (ly, top.cell_index (), db::recursive_cluster_shape_iterator<db::PolygonRef> (hc, l1, top.cell_index (), i->id ()));
    ++n;
  }
  EXPECT_EQ (n, 1);
  EXPECT_EQ (res, "TOP:(0,0;0,1000;1000,1000;1000,0);TOP/C1:(0,10;0,510;2000,510;2000,10);TOP/C2:(0,30;0,2030;500,2030;500,30);TOP/C2/C1:(0,50;0,550;2000,550;2000,50)");

  res.clear ();
  n = 0;
  cluster = &hc.clusters_per_cell (top.cell_index ());
  for (db::connected_clusters<db::PolygonRef>::const_iterator i = cluster->begin (); i != cluster->end (); ++i) {
    res = rcsiter2string (ly, top.cell_index (), db::recursive_cluster_shape_iterator<db::PolygonRef> (hc, l1, top.cell_index (), i->id ()), c1.cell_index ());
    ++n;
  }
  EXPECT_EQ (n, 1);
  EXPECT_EQ (res, "TOP:(0,0;0,1000;1000,1000;1000,0);TOP/C2:(0,30;0,2030;500,2030;500,30)");
}

TEST(41_HierClustersRecursiveClusterIterator)
{
  db::hier_clusters<db::PolygonRef> hc;

  db::Layout ly;
  unsigned int l1 = ly.insert_layer (db::LayerProperties (1, 0));

  db::Cell &top = ly.cell (ly.add_cell ("TOP"));
  top.shapes (l1).insert (make_box (ly, db::Box (0, 0, 1000, 1000)));

  db::Cell &c1 = ly.cell (ly.add_cell ("C1"));
  c1.shapes (l1).insert (make_box (ly, db::Box (0, 0, 2000, 500)));
  top.insert (db::CellInstArray (db::CellInst (c1.cell_index ()), db::Trans (db::Vector (0, 10))));

  db::Cell &c2 = ly.cell (ly.add_cell ("C2"));
  c2.shapes (l1).insert (make_box (ly, db::Box (0, 0, 500, 2000)));
  c2.insert (db::CellInstArray (db::CellInst (c1.cell_index ()), db::Trans (db::Vector (0, 20))));
  top.insert (db::CellInstArray (db::CellInst (c2.cell_index ()), db::Trans (db::Vector (0, 30))));

  db::Connectivity conn;
  conn.connect (l1, l1);

  hc.build (ly, top, conn);

  std::string res;
  int n = 0;
  db::connected_clusters<db::PolygonRef> *cluster = &hc.clusters_per_cell (top.cell_index ());
  for (db::connected_clusters<db::PolygonRef>::const_iterator i = cluster->begin (); i != cluster->end (); ++i) {
    res = rciter2string (ly, top.cell_index (), db::recursive_cluster_iterator<db::PolygonRef> (hc, top.cell_index (), i->id ()));
    ++n;
  }
  EXPECT_EQ (n, 1);
  EXPECT_EQ (res, "TOP;TOP/C1;TOP/C2;TOP/C2/C1");
}

static void normalize_layer (db::Layout &layout, std::vector<std::string> &strings, unsigned int &layer)
{
  unsigned int new_layer = layout.insert_layer ();

  for (db::Layout::iterator c = layout.begin (); c != layout.end (); ++c) {
    const db::Shapes &s = c->shapes (layer);
    for (db::Shapes::shape_iterator i = s.begin (db::ShapeIterator::Texts | db::ShapeIterator::Polygons | db::ShapeIterator::Paths | db::ShapeIterator::Boxes); !i.at_end (); ++i) {
      if (! i->is_text ()) {
        db::Polygon poly;
        i->polygon (poly);
        c->shapes (new_layer).insert (db::PolygonRef (poly, layout.shape_repository ()));
      } else {
        db::Polygon poly (i->bbox ());
        unsigned int attr_id = (unsigned int) strings.size () + 1;
        strings.push_back (i->text_string ());
        c->shapes (new_layer).insert (db::PolygonRefWithProperties (db::PolygonRef (poly, layout.shape_repository ()), attr_id));
      }
    }
  }

  layer = new_layer;
}

static void copy_cluster_shapes (const std::string *&attrs, db::Shapes &out, db::cell_index_type ci, const db::hier_clusters<db::PolygonRef> &hc, db::local_cluster<db::PolygonRef>::id_type cluster_id, const db::ICplxTrans &trans, const db::Connectivity &conn)
{
  //  use property #1 to code the cell name
  //  use property #2 to code the attrs string for the first shape

  db::properties_id_type cell_pid = 0, cell_and_attr_pid = 0;

  db::PropertiesSet pm;
  pm.insert (tl::Variant (1), tl::Variant (out.layout ()->cell_name (ci)));
  cell_pid = db::properties_id (pm);

  if (attrs && ! attrs->empty ()) {
    pm.insert (tl::Variant (2), tl::Variant (*attrs));
    cell_and_attr_pid = db::properties_id (pm);
  }

  const db::connected_clusters<db::PolygonRef> &clusters = hc.clusters_per_cell (ci);
  const db::local_cluster<db::PolygonRef> &lc = clusters.cluster_by_id (cluster_id);

  //  copy the shapes from this cell
  for (db::Connectivity::all_layer_iterator l = conn.begin_layers (); l != conn.end_layers (); ++l) {
    for (db::local_cluster<db::PolygonRef>::shape_iterator s = lc.begin (*l); ! s.at_end (); ++s) {
      db::Polygon poly = s->obj ().transformed (trans * db::ICplxTrans (s->trans ()));
      out.insert (db::PolygonWithProperties (poly, cell_and_attr_pid > 0 ? cell_and_attr_pid : cell_pid));
      cell_and_attr_pid = 0;
      attrs = 0; // used
    }
  }

  out.layout ()->cell_name (ci);

  //  copy the shapes from the child cells too
  typedef db::connected_clusters<db::PolygonRef>::connections_type connections_type;
  const connections_type &connections = clusters.connections_for_cluster (cluster_id);
  for (connections_type::const_iterator i = connections.begin (); i != connections.end (); ++i) {

    db::ICplxTrans t = trans * i->inst_trans ();

    db::cell_index_type cci = i->inst_cell_index ();
    copy_cluster_shapes (attrs, out, cci, hc, i->id (), t, conn);

  }
}

static void run_hc_test (tl::TestBase *_this, const std::string &file, const std::string &au_file)
{
  db::Layout ly;
  unsigned int l1 = 0, l2 = 0, l3 = 0, l4 = 0, l5 = 0, l6 = 0;

  {
    db::LayerProperties p;
    db::LayerMap lmap;

    p.layer = 1;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l1 = ly.insert_layer ());
    ly.set_properties (l1, p);

    p.layer = 2;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l2 = ly.insert_layer ());
    ly.set_properties (l2, p);

    p.layer = 3;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l3 = ly.insert_layer ());
    ly.set_properties (l3, p);

    p.layer = 4;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l4 = ly.insert_layer ());
    ly.set_properties (l4, p);

    p.layer = 5;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l5 = ly.insert_layer ());
    ly.set_properties (l5, p);

    p.layer = 6;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l6 = ly.insert_layer ());
    ly.set_properties (l6, p);

    db::LoadLayoutOptions options;
    options.get_options<db::CommonReaderOptions> ().layer_map = lmap;
    options.get_options<db::CommonReaderOptions> ().create_other_layers = false;

    std::string fn (tl::testdata ());
    fn += "/algo/";
    fn += file;
    tl::InputStream stream (fn);
    db::Reader reader (stream);
    reader.read (ly, options);
  }

  std::vector<std::string> strings;
  normalize_layer (ly, strings, l1);
  normalize_layer (ly, strings, l2);
  normalize_layer (ly, strings, l3);
  normalize_layer (ly, strings, l4);
  normalize_layer (ly, strings, l5);
  normalize_layer (ly, strings, l6);

  //  connect 1 to 1, 1 to 2 and 1 to 3, but *not* 2 to 3
  db::Connectivity conn;
  conn.connect (l1, l1);
  conn.connect (l2, l2);
  conn.connect (l3, l3);
  conn.connect (l1, l2);
  conn.connect (l1, l3);
  conn.connect (l1, l4);
  conn.connect (l1, l5);
  conn.connect (l1, l6);

  conn.connect_global (l4, "BULK");
  conn.connect_global (l5, "BULK2");
  conn.connect_global (l6, "BULK");
  conn.connect_global (l6, "BULK2");

  db::hier_clusters<db::PolygonRef> hc;
  hc.build (ly, ly.cell (*ly.begin_top_down ()), conn);

  std::vector<std::pair<db::Polygon::area_type, unsigned int> > net_layers;

  for (db::Layout::top_down_const_iterator td = ly.begin_top_down (); td != ly.end_top_down (); ++td) {

    const db::connected_clusters<db::PolygonRef> &clusters = hc.clusters_per_cell (*td);
    for (db::connected_clusters<db::PolygonRef>::all_iterator c = clusters.begin_all (); ! c.at_end (); ++c) {

      if (! clusters.is_root (*c)) {
        continue;
      }

      //  collect strings
      std::string attrs;
      for (db::recursive_cluster_iterator<db::PolygonRef> rc (hc, *td, *c); ! rc.at_end (); ++rc) {
        const db::local_cluster<db::PolygonRef> &rcc = hc.clusters_per_cell (rc.cell_index ()).cluster_by_id (rc.cluster_id ());
        for (db::local_cluster<db::PolygonRef>::attr_iterator a = rcc.begin_attr (); a != rcc.end_attr (); ++a) {
          if (! attrs.empty ()) {
            attrs += "/";
          }
          attrs += std::string (ly.cell_name (rc.cell_index ())) + ":" + strings[*a - 1];
        }
      }

      net_layers.push_back (std::make_pair (0, ly.insert_layer ()));

      unsigned int lout = net_layers.back ().second;

      db::Shapes &out = ly.cell (*td).shapes (lout);
      const std::string *attrs_str = &attrs;
      copy_cluster_shapes (attrs_str, out, *td, hc, *c, db::ICplxTrans (), conn);

      db::Polygon::area_type area = 0;
      for (db::Shapes::shape_iterator s = out.begin (db::ShapeIterator::All); ! s.at_end (); ++s) {
        area += s->area ();
      }
      net_layers.back ().first = area;

    }

  }

  //  sort layers by area so we have a consistent numbering
  std::sort (net_layers.begin (), net_layers.end ());
  std::reverse (net_layers.begin (), net_layers.end ());

  int ln = 1000;
  for (std::vector<std::pair<db::Polygon::area_type, unsigned int> >::const_iterator l = net_layers.begin (); l != net_layers.end (); ++l) {
    ly.set_properties (l->second, db::LayerProperties (ln, 0));
    ++ln;
  }

  CHECKPOINT();
  db::compare_layouts (_this, ly, tl::testdata () + "/algo/" + au_file);
}

static void run_hc_test_with_backannotation (tl::TestBase *_this, const std::string &file, const std::string &au_file)
{
  db::Layout ly;
  unsigned int l1 = 0, l2 = 0, l3 = 0, l4 = 0, l5 = 0, l6 = 0;

  {
    db::LayerProperties p;
    db::LayerMap lmap;

    p.layer = 1;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l1 = ly.insert_layer ());
    ly.set_properties (l1, p);

    p.layer = 2;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l2 = ly.insert_layer ());
    ly.set_properties (l2, p);

    p.layer = 3;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l3 = ly.insert_layer ());
    ly.set_properties (l3, p);

    p.layer = 4;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l4 = ly.insert_layer ());
    ly.set_properties (l4, p);

    p.layer = 5;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l5 = ly.insert_layer ());
    ly.set_properties (l5, p);

    p.layer = 6;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l6 = ly.insert_layer ());
    ly.set_properties (l6, p);

    db::LoadLayoutOptions options;
    options.get_options<db::CommonReaderOptions> ().layer_map = lmap;
    options.get_options<db::CommonReaderOptions> ().create_other_layers = false;

    std::string fn (tl::testdata ());
    fn += "/algo/";
    fn += file;
    tl::InputStream stream (fn);
    db::Reader reader (stream);
    reader.read (ly, options);
  }

  std::vector<std::string> strings;
  normalize_layer (ly, strings, l1);
  normalize_layer (ly, strings, l2);
  normalize_layer (ly, strings, l3);
  normalize_layer (ly, strings, l4);
  normalize_layer (ly, strings, l5);
  normalize_layer (ly, strings, l6);

  //  connect 1 to 1, 1 to 2 and 1 to 3, but *not* 2 to 3
  db::Connectivity conn;
  conn.connect (l1, l1);
  conn.connect (l2, l2);
  conn.connect (l3, l3);
  conn.connect (l1, l2);
  conn.connect (l1, l3);
  conn.connect (l1, l4);
  conn.connect (l1, l5);
  conn.connect (l1, l6);

  conn.connect_global (l4, "BULK");
  conn.connect_global (l5, "BULK2");
  conn.connect_global (l6, "BULK");
  conn.connect_global (l6, "BULK2");

  db::hier_clusters<db::PolygonRef> hc;
  hc.build (ly, ly.cell (*ly.begin_top_down ()), conn);

  std::map<unsigned int, unsigned int> lm;
  lm[l1] = ly.insert_layer (db::LayerProperties (101, 0));
  lm[l2] = ly.insert_layer (db::LayerProperties (102, 0));
  lm[l3] = ly.insert_layer (db::LayerProperties (103, 0));
  lm[l4] = ly.insert_layer (db::LayerProperties (104, 0));
  lm[l5] = ly.insert_layer (db::LayerProperties (105, 0));
  lm[l6] = ly.insert_layer (db::LayerProperties (106, 0));
  hc.return_to_hierarchy (ly, lm);

  CHECKPOINT();
  db::compare_layouts (_this, ly, tl::testdata () + "/algo/" + au_file);
}

TEST(101_HierClusters)
{
  run_hc_test (_this, "hc_test_l1.gds", "hc_test_au1.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l1.gds", "hc_test_au1b.gds");
}

TEST(102_HierClusters)
{
  run_hc_test (_this, "hc_test_l2.gds", "hc_test_au2.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l2.gds", "hc_test_au2b.gds");
}

TEST(103_HierClusters)
{
  run_hc_test (_this, "hc_test_l3.gds", "hc_test_au3.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l3.gds", "hc_test_au3b.gds");
}

TEST(104_HierClusters)
{
  run_hc_test (_this, "hc_test_l4.gds", "hc_test_au4.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l4.gds", "hc_test_au4b.gds");
}

TEST(105_HierClusters)
{
  run_hc_test (_this, "hc_test_l5.gds", "hc_test_au5.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l5.gds", "hc_test_au5b.gds");
}

TEST(106_HierClusters)
{
  run_hc_test (_this, "hc_test_l6.gds", "hc_test_au6.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l6.gds", "hc_test_au6b.gds");
}

TEST(107_HierClusters)
{
  run_hc_test (_this, "hc_test_l7.gds", "hc_test_au7.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l7.gds", "hc_test_au7b.gds");
}

TEST(108_HierClusters)
{
  run_hc_test (_this, "hc_test_l8.gds", "hc_test_au8.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l8.gds", "hc_test_au8b.gds");
}

TEST(109_HierClusters)
{
  run_hc_test (_this, "hc_test_l9.gds", "hc_test_au9.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l9.gds", "hc_test_au9b.gds");
}

TEST(110_HierClusters)
{
  run_hc_test (_this, "hc_test_l10.gds", "hc_test_au10.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l10.gds", "hc_test_au10b.gds");
}

TEST(111_HierClusters)
{
  run_hc_test (_this, "hc_test_l11.gds", "hc_test_au11.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l11.gds", "hc_test_au11b.gds");
}

TEST(112_HierClusters)
{
  run_hc_test (_this, "hc_test_l12.gds", "hc_test_au12.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l12.gds", "hc_test_au12b.gds");
}

TEST(113_HierClusters)
{
  run_hc_test (_this, "hc_test_l13.gds", "hc_test_au13.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l13.gds", "hc_test_au13b.gds");
}

TEST(114_HierClusters)
{
  run_hc_test (_this, "hc_test_l14.gds", "hc_test_au14.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l14.gds", "hc_test_au14b.gds");
}

TEST(115_HierClusters)
{
  run_hc_test (_this, "hc_test_l15.gds", "hc_test_au15.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l15.gds", "hc_test_au15b.gds");
}

TEST(116_HierClusters)
{
  run_hc_test (_this, "hc_test_l16.gds", "hc_test_au16.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l16.gds", "hc_test_au16b.gds");
}

TEST(117_HierClusters)
{
  run_hc_test (_this, "hc_test_l17.gds", "hc_test_au17.gds");
  run_hc_test_with_backannotation (_this, "hc_test_l17.gds", "hc_test_au17b.gds");
}

TEST(118_HierClustersMeanderArrays)
{
  run_hc_test (_this, "meander.gds.gz", "meander_au1.gds");
  run_hc_test_with_backannotation (_this, "meander.gds.gz", "meander_au2.gds");
}

TEST(119_HierClustersCombArrays)
{
  run_hc_test (_this, "comb.gds", "comb_au1.gds");
  run_hc_test_with_backannotation (_this, "comb.gds", "comb_au2.gds");
}

TEST(120_HierClustersCombArrays)
{
  run_hc_test (_this, "comb2.gds", "comb2_au1.gds");
  run_hc_test_with_backannotation (_this, "comb2.gds", "comb2_au2.gds");
}

static size_t root_nets (const db::connected_clusters<db::PolygonRef> &cc)
{
  size_t n = 0;
  for (db::connected_clusters<db::PolygonRef>::all_iterator c = cc.begin_all (); !c.at_end (); ++c) {
    if (cc.is_root (*c)) {
      ++n;
    }
  }
  return n;
}

class ScopedEnvironment
{
public:
  ScopedEnvironment (const std::string &name, const std::string &value)
    : m_name (name), m_was_set (tl::has_env (name)),
      m_old_value (m_was_set ? tl::get_env (name) : std::string ())
  {
    tl::set_env (m_name, value);
  }

  ~ScopedEnvironment ()
  {
    if (m_was_set) {
      tl::set_env (m_name, m_old_value);
    } else {
      tl::unset_env (m_name);
    }
  }

private:
  ScopedEnvironment (const ScopedEnvironment &);
  ScopedEnvironment &operator= (const ScopedEnvironment &);

  std::string m_name;
  bool m_was_set;
  std::string m_old_value;
};

static std::string
cluster_path_signature (
  const db::Layout &ly, db::cell_index_type ci,
  const std::vector<db::ClusterInstance> &path)
{
  std::string signature = ly.cell_name (ci);
  for (std::vector<db::ClusterInstance>::const_iterator p = path.begin ();
       p != path.end (); ++p) {
    signature += "/" + std::string (ly.cell_name (p->inst_cell_index ()));
    signature += "@" + p->inst_trans ().to_string ();
    signature += "#" + tl::to_string (p->inst_prop_id ());
  }
  return signature;
}

static std::string
cluster_attrs_signature (
  const db::local_cluster<db::PolygonRef> &cluster)
{
  std::string signature = "attrs{";
  for (db::local_cluster<db::PolygonRef>::attr_iterator a =
         cluster.begin_attr (); a != cluster.end_attr (); ++a) {
    signature += tl::to_string (*a) + ",";
  }
  signature += "}";
  return signature;
}

static std::string
hierarchy_signature (const db::Layout &ly,
                     const db::hier_clusters<db::PolygonRef> &hc,
                     unsigned int layer)
{
  std::vector<std::string> cells;

  for (db::Layout::const_iterator c = ly.begin (); c != ly.end (); ++c) {
    const db::cell_index_type ci = c->cell_index ();
    const db::connected_clusters<db::PolygonRef> &cc = hc.clusters_per_cell (ci);
    std::vector<std::string> roots;

    for (db::connected_clusters<db::PolygonRef>::all_iterator r = cc.begin_all (); ! r.at_end (); ++r) {
      if (! cc.is_root (*r)) {
        continue;
      }

      std::vector<std::string> shapes;
      db::recursive_cluster_shape_iterator<db::PolygonRef> si (hc, layer, ci, *r);
      while (! si.at_end ()) {
        db::Polygon poly = si->obj ();
        poly.transform (si->trans ());
        poly.transform (si.trans ());
        const db::local_cluster<db::PolygonRef> &shape_cluster =
          hc.clusters_per_cell (si.cell_index ()).cluster_by_id (
            si.cluster_id ());
        shapes.push_back (
          cluster_path_signature (ly, ci, si.inst_path ()) + ":" +
          cluster_attrs_signature (shape_cluster) + ":" + poly.to_string ());
        ++si;
      }
      std::sort (shapes.begin (), shapes.end ());

      const db::local_cluster<db::PolygonRef> &cluster =
        cc.cluster_by_id (*r);
      std::string root = cluster_attrs_signature (cluster) +
                         "shapes" + tl::to_string (shapes.size ()) + "{";
      for (std::vector<std::string>::const_iterator s = shapes.begin (); s != shapes.end (); ++s) {
        root += tl::to_string (s->size ()) + ":" + *s;
      }
      root += "}";
      roots.push_back (root);
    }

    std::sort (roots.begin (), roots.end ());
    std::string cell = std::string (ly.cell_name (ci)) + "#" + tl::to_string (roots.size ()) + "[";
    for (std::vector<std::string>::const_iterator r = roots.begin (); r != roots.end (); ++r) {
      cell += tl::to_string (r->size ()) + ":" + *r;
    }
    cell += "]";
    cells.push_back (cell);
  }

  std::sort (cells.begin (), cells.end ());
  std::string signature;
  for (std::vector<std::string>::const_iterator c = cells.begin (); c != cells.end (); ++c) {
    signature += tl::to_string (c->size ()) + ":" + *c;
  }
  return signature;
}

TEST(121_HierClustersIndependentComponents)
{
  db::Layout ly;
  unsigned int l1 = ly.insert_layer (db::LayerProperties (1, 0));

  db::Cell &top = ly.cell (ly.add_cell ("TOP"));
  db::Cell &a = ly.cell (ly.add_cell ("A"));
  db::Cell &la = ly.cell (ly.add_cell ("LA"));
  db::Cell &b = ly.cell (ly.add_cell ("B"));
  db::Cell &lb = ly.cell (ly.add_cell ("LB"));

  la.shapes (l1).insert (make_box (ly, db::Box (0, 0, 100, 100)));
  a.shapes (l1).insert (make_box (ly, db::Box (50, 0, 160, 100)));
  a.insert (db::CellInstArray (db::CellInst (la.cell_index ()), db::Trans ()));
  top.shapes (l1).insert (make_box (ly, db::Box (100, 0, 180, 100)));
  top.insert (db::CellInstArray (db::CellInst (a.cell_index ()), db::Trans ()));

  lb.shapes (l1).insert (make_box (ly, db::Box (20, 0, 120, 100)));
  b.shapes (l1).insert (make_box (ly, db::Box (0, 0, 70, 100)));
  b.insert (db::CellInstArray (db::CellInst (lb.cell_index ()), db::Trans ()));
  top.shapes (l1).insert (make_box (ly, db::Box (1000, 0, 1060, 100)));
  top.insert (db::CellInstArray (db::CellInst (b.cell_index ()),
                                db::Trans (db::Vector (1000, 0))));
  //  Join both worker-owned cones only at the serial top boundary.
  top.shapes (l1).insert (make_box (ly, db::Box (150, 0, 1020, 100)));

  db::Connectivity conn;
  conn.connect (l1, l1);

  db::hier_clusters<db::PolygonRef> serial;
  serial.build (ly, top, conn, 0, 0, false, 1u);
  db::hier_clusters<db::PolygonRef> parallel;
  std::string telemetry;
  {
    ScopedEnvironment enabled (
      "KLAYOUT_HIER_NETWORK_COMPONENTS_TELEMETRY", "1");
    tl::CaptureChannel capture;
    parallel.build (ly, top, conn, 0, 0, false, 2u);
    telemetry = capture.captured_text ();
  }

  EXPECT_EQ (telemetry.find ("outcome=parallel") != std::string::npos, true);
  EXPECT_EQ (root_nets (serial.clusters_per_cell (top.cell_index ())), size_t (1));
  EXPECT_EQ (hierarchy_signature (ly, serial, l1),
             hierarchy_signature (ly, parallel, l1));
}

TEST(122_HierClustersSharedDescendantFallback)
{
  db::Layout ly;
  unsigned int l1 = ly.insert_layer (db::LayerProperties (1, 0));

  db::Cell &top = ly.cell (ly.add_cell ("TOP"));
  db::Cell &a = ly.cell (ly.add_cell ("A"));
  db::Cell &b = ly.cell (ly.add_cell ("B"));
  db::Cell &shared = ly.cell (ly.add_cell ("SHARED"));

  shared.shapes (l1).insert (make_box (ly, db::Box (0, 0, 100, 100)));
  a.shapes (l1).insert (make_box (ly, db::Box (40, 0, 140, 100)));
  b.shapes (l1).insert (make_box (ly, db::Box (20, 0, 120, 100)));
  a.insert (db::CellInstArray (db::CellInst (shared.cell_index ()), db::Trans ()));
  b.insert (db::CellInstArray (db::CellInst (shared.cell_index ()), db::Trans ()));
  top.shapes (l1).insert (make_box (ly, db::Box (80, 0, 160, 100)));
  top.shapes (l1).insert (make_box (ly, db::Box (1080, 0, 1160, 100)));
  top.insert (db::CellInstArray (db::CellInst (a.cell_index ()), db::Trans ()));
  top.insert (db::CellInstArray (db::CellInst (b.cell_index ()),
                                db::Trans (db::Vector (1000, 0))));

  db::Connectivity conn;
  conn.connect (l1, l1);

  db::hier_clusters<db::PolygonRef> serial;
  serial.build (ly, top, conn, 0, 0, false, 1u);
  db::hier_clusters<db::PolygonRef> requested_parallel;
  std::string telemetry;
  {
    ScopedEnvironment enabled (
      "KLAYOUT_HIER_NETWORK_COMPONENTS_TELEMETRY", "1");
    tl::CaptureChannel capture;
    requested_parallel.build (ly, top, conn, 0, 0, false, 2u);
    telemetry = capture.captured_text ();
  }

  EXPECT_EQ (telemetry.find ("reason=shared-descendant") != std::string::npos,
             true);
  EXPECT_EQ (root_nets (serial.clusters_per_cell (top.cell_index ())), size_t (2));
  EXPECT_EQ (hierarchy_signature (ly, serial, l1),
             hierarchy_signature (ly, requested_parallel, l1));
}

TEST(123_HierClustersExternalParentFallback)
{
  db::Layout ly;
  unsigned int l1 = ly.insert_layer (db::LayerProperties (1, 0));

  db::Cell &selected = ly.cell (ly.add_cell ("SELECTED"));
  db::Cell &a = ly.cell (ly.add_cell ("A"));
  db::Cell &la = ly.cell (ly.add_cell ("LA"));
  db::Cell &b = ly.cell (ly.add_cell ("B"));
  db::Cell &lb = ly.cell (ly.add_cell ("LB"));
  db::Cell &outside = ly.cell (ly.add_cell ("OUTSIDE"));

  la.shapes (l1).insert (make_box (ly, db::Box (0, 0, 100, 100)));
  a.shapes (l1).insert (make_box (ly, db::Box (40, 0, 140, 100)));
  a.insert (db::CellInstArray (db::CellInst (la.cell_index ()), db::Trans ()));
  selected.shapes (l1).insert (make_box (ly, db::Box (80, 0, 160, 100)));
  selected.insert (db::CellInstArray (db::CellInst (a.cell_index ()), db::Trans ()));

  lb.shapes (l1).insert (make_box (ly, db::Box (0, 0, 80, 100)));
  b.shapes (l1).insert (make_box (ly, db::Box (20, 0, 120, 100)));
  b.insert (db::CellInstArray (db::CellInst (lb.cell_index ()), db::Trans ()));
  selected.shapes (l1).insert (make_box (ly, db::Box (1020, 0, 1100, 100)));
  selected.insert (db::CellInstArray (db::CellInst (b.cell_index ()),
                                     db::Trans (db::Vector (1000, 0))));

  outside.insert (db::CellInstArray (db::CellInst (la.cell_index ()),
                                    db::Trans (db::Vector (2000, 0))));

  db::Connectivity conn;
  conn.connect (l1, l1);

  db::hier_clusters<db::PolygonRef> serial;
  serial.build (ly, selected, conn, 0, 0, false, 1u);
  db::hier_clusters<db::PolygonRef> requested_parallel;
  std::string telemetry;
  {
    ScopedEnvironment enabled (
      "KLAYOUT_HIER_NETWORK_COMPONENTS_TELEMETRY", "1");
    tl::CaptureChannel capture;
    requested_parallel.build (ly, selected, conn, 0, 0, false, 2u);
    telemetry = capture.captured_text ();
  }

  EXPECT_EQ (
    telemetry.find ("reason=parent-outside-component") != std::string::npos,
    true);
  EXPECT_EQ (root_nets (serial.clusters_per_cell (selected.cell_index ())), size_t (2));
  EXPECT_EQ (root_nets (serial.clusters_per_cell (outside.cell_index ())), size_t (1));
  EXPECT_EQ (hierarchy_signature (ly, serial, l1),
             hierarchy_signature (ly, requested_parallel, l1));
}

TEST(124_HierClustersIndependentComponentsSeparateAttributes)
{
  db::Layout ly;
  unsigned int l1 = ly.insert_layer (db::LayerProperties (1, 0));

  db::PropertiesSet props1;
  props1.insert (tl::Variant ("net"), tl::Variant ("one"));
  const db::properties_id_type pid1 = db::properties_id (props1);
  db::PropertiesSet props2;
  props2.insert (tl::Variant ("net"), tl::Variant ("two"));
  const db::properties_id_type pid2 = db::properties_id (props2);

  db::Cell &top = ly.cell (ly.add_cell ("TOP"));
  db::Cell &a = ly.cell (ly.add_cell ("A"));
  db::Cell &la = ly.cell (ly.add_cell ("LA"));
  db::Cell &b = ly.cell (ly.add_cell ("B"));
  db::Cell &lb = ly.cell (ly.add_cell ("LB"));

  la.shapes (l1).insert (db::PolygonRefWithProperties (
    make_box (ly, db::Box (0, 0, 100, 100)), pid1));
  a.shapes (l1).insert (db::PolygonRefWithProperties (
    make_box (ly, db::Box (50, 0, 150, 100)), pid1));
  a.insert (db::CellInstArray (db::CellInst (la.cell_index ()), db::Trans ()));

  lb.shapes (l1).insert (db::PolygonRefWithProperties (
    make_box (ly, db::Box (0, 0, 100, 100)), pid2));
  b.shapes (l1).insert (db::PolygonRefWithProperties (
    make_box (ly, db::Box (50, 0, 150, 100)), pid2));
  b.insert (db::CellInstArray (db::CellInst (lb.cell_index ()), db::Trans ()));

  //  Both cones occupy the same top-level window.  Property separation must
  //  keep them distinct even though each matching top shape touches both.
  top.insert (db::CellInstArray (db::CellInst (a.cell_index ()), db::Trans ()));
  top.insert (db::CellInstArray (db::CellInst (b.cell_index ()), db::Trans ()));
  top.shapes (l1).insert (db::PolygonRefWithProperties (
    make_box (ly, db::Box (100, 0, 180, 100)), pid1));
  top.shapes (l1).insert (db::PolygonRefWithProperties (
    make_box (ly, db::Box (100, 0, 180, 100)), pid2));

  db::Connectivity conn;
  conn.connect (l1, l1);

  db::hier_clusters<db::PolygonRef> attributes_joined;
  attributes_joined.build (ly, top, conn, 0, 0, false, 1u);
  db::hier_clusters<db::PolygonRef> serial;
  serial.build (ly, top, conn, 0, 0, true, 1u);
  db::hier_clusters<db::PolygonRef> parallel;
  std::string telemetry;
  {
    ScopedEnvironment enabled (
      "KLAYOUT_HIER_NETWORK_COMPONENTS_TELEMETRY", "1");
    tl::CaptureChannel capture;
    parallel.build (ly, top, conn, 0, 0, true, 2u);
    telemetry = capture.captured_text ();
  }

  EXPECT_EQ (telemetry.find ("outcome=parallel") != std::string::npos, true);
  EXPECT_EQ (
    root_nets (attributes_joined.clusters_per_cell (top.cell_index ())),
    size_t (1));
  EXPECT_EQ (root_nets (serial.clusters_per_cell (top.cell_index ())), size_t (2));
  EXPECT_EQ (hierarchy_signature (ly, serial, l1),
             hierarchy_signature (ly, parallel, l1));
}

//  issue #609
TEST(200_issue609)
{
  db::Layout ly;
  unsigned int l1 = 0, l2 = 0;

  {
    db::LayerProperties p;
    db::LayerMap lmap;

    p.layer = 1;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l1 = ly.insert_layer ());
    ly.set_properties (l1, p);

    p.layer = 2;
    p.datatype = 0;
    lmap.map (db::LDPair (p.layer, p.datatype), l2 = ly.insert_layer ());
    ly.set_properties (l2, p);

    db::LoadLayoutOptions options;
    options.get_options<db::CommonReaderOptions> ().layer_map = lmap;
    options.get_options<db::CommonReaderOptions> ().create_other_layers = false;

    std::string fn (tl::testdata ());
    fn += "/algo/issue-609.oas.gz";
    tl::InputStream stream (fn);
    db::Reader reader (stream);
    reader.read (ly, options);
  }

  std::vector<std::string> strings;
  normalize_layer (ly, strings, l1);
  normalize_layer (ly, strings, l2);

  //  connect 1 to 1, 1 to 2
  db::Connectivity conn;
  conn.connect (l1, l1);
  conn.connect (l2, l2);
  conn.connect (l1, l2);

  db::hier_clusters<db::PolygonRef> hc;
  hc.build (ly, ly.cell (*ly.begin_top_down ()), conn);

  db::Layout::top_down_const_iterator td = ly.begin_top_down ();
  EXPECT_EQ (td != ly.end_top_down (), true);
  EXPECT_EQ (root_nets (hc.clusters_per_cell (*td)), size_t (1));
  ++td;

  //  result needs to be a single net
  for ( ; td != ly.end_top_down (); ++td) {
    EXPECT_EQ (root_nets (hc.clusters_per_cell (*td)), size_t (0));
  }
}

//  issue #1126
TEST(201_issue1126)
{
  {
    db::Layout ly;
    unsigned int l1 = 0;

    {
      db::LayerProperties p;
      db::LayerMap lmap;

      p.layer = 1;
      p.datatype = 0;
      lmap.map (db::LDPair (p.layer, p.datatype), l1 = ly.insert_layer ());
      ly.set_properties (l1, p);

      db::LoadLayoutOptions options;
      options.get_options<db::CommonReaderOptions> ().layer_map = lmap;
      options.get_options<db::CommonReaderOptions> ().create_other_layers = false;

      std::string fn (tl::testdata ());
      fn += "/algo/issue-1126.gds.gz";
      tl::InputStream stream (fn);
      db::Reader reader (stream);
      reader.read (ly, options);
    }

    std::vector<std::string> strings;
    normalize_layer (ly, strings, l1);

    //  connect 1 to 1
    db::Connectivity conn;
    conn.connect (l1, l1);

    db::hier_clusters<db::PolygonRef> hc;
    hc.build (ly, ly.cell (*ly.begin_top_down ()), conn);

    // should not assert until here
  }

  //  detailed test:
  run_hc_test (_this, "issue-1126.gds.gz", "issue-1126_au.gds");
}
