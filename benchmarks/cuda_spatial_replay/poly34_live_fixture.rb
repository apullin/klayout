# frozen_string_literal: true

# Focused hierarchy and positive-area fixtures for the atomic live
# FreePDK45 POLY.3/POLY.4 terminal-empty transaction.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005
active = layout.layer(1, 0)
poly = layout.layer(9, 0)
wrong_poly = layout.layer(19, 0)

def poly34_cell(layout, name, poly_layer, active_layer, poly_boxes, active_boxes)
  cell = layout.create_cell(name)
  poly_boxes.each { |box| cell.shapes(poly_layer).insert(box) }
  active_boxes.each { |box| cell.shapes(active_layer).insert(box) }
  cell
end

gate = RBA::Box.new(0, 0, 100, 180)

# Both historical enclosing checks form only coincident zero-area polygons.
clean = poly34_cell(layout, "POLY34_CLEAN", poly, active, [gate], [gate])
# Duplicate only this fixture's POLY geometry on an unqualified layer.  The
# wrong-layer live lane reads 19/0 and must decline before scene lowering while
# producing the same pristine CPU report.
clean.shapes(wrong_poly).insert(gate)

# Disjoint rectilinear non-box components exercise exact TD_simple
# decomposition for both primary domains while leaving the derived GATE as the
# same clean rectangle.
manhattan_poly = RBA::Polygon.new(
  [
    RBA::Point.new(10_000, 10_000),
    RBA::Point.new(10_400, 10_000),
    RBA::Point.new(10_400, 10_100),
    RBA::Point.new(10_100, 10_100),
    RBA::Point.new(10_100, 10_400),
    RBA::Point.new(10_000, 10_400)
  ]
)
manhattan_active = RBA::Polygon.new(
  [
    RBA::Point.new(20_000, 20_000),
    RBA::Point.new(20_500, 20_000),
    RBA::Point.new(20_500, 20_100),
    RBA::Point.new(20_100, 20_100),
    RBA::Point.new(20_100, 20_500),
    RBA::Point.new(20_000, 20_500)
  ]
)
manhattan_clean = poly34_cell(
  layout,
  "POLY34_MANHATTAN_PRIMARY_CLEAN",
  poly,
  active,
  [gate],
  [gate]
)
manhattan_clean.shapes(poly).insert(manhattan_poly)
manhattan_clean.shapes(active).insert(manhattan_active)

# A one-DBU partial extension creates a genuine positive-area POLY.3 marker
# while POLY.4 remains coincident/terminal-empty.
poly34_cell(
  layout,
  "POLY34_POLY3_HIT",
  poly,
  active,
  [RBA::Box.new(-1, 0, 100, 180)],
  [gate]
)

# The independent one-DBU ACTIVE extension creates a genuine POLY.4 marker.
poly34_cell(
  layout,
  "POLY34_POLY4_HIT",
  poly,
  active,
  [gate],
  [RBA::Box.new(0, -1, 100, 180)]
)

# Two distant hierarchy leaves exercise a mixed transaction: one gate fails
# only POLY.3 and the other fails only POLY.4.  No partial rule result may be
# consumed; both complete historical expressions must publish their markers.
mixed_poly3 = poly34_cell(
  layout,
  "POLY34_MIXED_POLY3_LEAF",
  poly,
  active,
  [RBA::Box.new(-1, 0, 100, 180)],
  [gate]
)
mixed_poly4 = poly34_cell(
  layout,
  "POLY34_MIXED_POLY4_LEAF",
  poly,
  active,
  [gate],
  [RBA::Box.new(0, -1, 100, 180)]
)
mixed = layout.create_cell("POLY34_MIXED")
mixed.insert(
  RBA::CellInstArray.new(
    mixed_poly3.cell_index,
    RBA::Trans.new(RBA::Trans::R90, 5_000, 5_000)
  )
)
mixed.insert(
  RBA::CellInstArray.new(
    mixed_poly4.cell_index,
    RBA::Trans.new(RBA::Trans::M90, 15_000, 5_000)
  )
)

# A dedicated asymmetric, rotated hierarchy hit ensures a positive-area result
# depends on transforming both the leaf geometry and non-square array vectors.
hier_transform_hit = layout.create_cell("POLY34_HIER_TRANSFORM_HIT")
hier_transform_hit.insert(
  RBA::CellInstArray.new(
    mixed_poly3.cell_index,
    RBA::Trans.new(RBA::Trans::R270, 30_000, 40_000),
    RBA::Vector.new(2_000, 0),
    RBA::Vector.new(0, 3_000),
    3,
    2
  )
)

# The leaf template is stored once and occurs 2048 times through two
# transformed 32x32 arrays.  Every gate is clean by the coincident zero-area
# terminal rule.  This is the hierarchy-reuse head check, not a flat census.
hier_leaf = poly34_cell(
  layout,
  "POLY34_HIER_CLEAN_LEAF",
  poly,
  active,
  [gate],
  [gate]
)
hier_array = layout.create_cell("POLY34_HIER_CLEAN_ARRAY")
hier_array.insert(
  RBA::CellInstArray.new(
    hier_leaf.cell_index,
    RBA::Trans.new,
    RBA::Vector.new(1_000, 0),
    RBA::Vector.new(0, 1_000),
    32,
    32
  )
)
hier_clean = layout.create_cell("POLY34_HIER_CLEAN")
hier_clean.insert(
  RBA::CellInstArray.new(
    hier_array.cell_index,
    RBA::Trans.new(RBA::Trans::R90, 50_000, 50_000)
  )
)
hier_clean.insert(
  RBA::CellInstArray.new(
    hier_array.cell_index,
    RBA::Trans.new(RBA::Trans::M90, 150_000, 50_000)
  )
)

expected_tops = %w[
  POLY34_CLEAN
  POLY34_HIER_CLEAN
  POLY34_HIER_TRANSFORM_HIT
  POLY34_MANHATTAN_PRIMARY_CLEAN
  POLY34_MIXED
  POLY34_POLY3_HIT
  POLY34_POLY4_HIT
].sort
actual_tops = layout.top_cells.map(&:name).sort
unless actual_tops == expected_tops
  raise("unexpected top cells: #{actual_tops.join(',')}")
end

layout.write(output)
puts(
  "POLY34_LIVE_FIXTURE ok path=#{output}" \
  " tops=#{expected_tops.length} dbu=#{layout.dbu}" \
  " hierarchy_gate_occurrences=2048 transformed_hit_occurrences=6"
)
