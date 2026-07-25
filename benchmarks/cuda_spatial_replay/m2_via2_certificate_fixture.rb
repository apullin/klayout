# frozen_string_literal: true

# Deterministic METAL2.4 fixtures. Layer 101/0 is raw M2 and layer 102/0 is
# raw VIA2 so each named top can be exported directly to KACTSCN1.

include RBA

def fixture_error(message)
  raise("M2/VIA2 certificate fixture: #{message}")
end

def required_rd(name, value)
  fixture_error("missing -rd #{name}=VALUE") if value.nil? || value.to_s.empty?
  value.to_s
end

def margin_case(layout, metal_layer, via_layer, name, left, right, bottom, top)
  cell = layout.create_cell(name)
  cell.shapes(metal_layer).insert(
    RBA::Box.new(100 - left, 100 - bottom, 230 + right, 230 + top)
  )
  cell.shapes(via_layer).insert(RBA::Box.new(100, 100, 230, 230))
  cell
end

def insert_tiled_x(cell, metal_layer, gap = false)
  cell.shapes(metal_layer).insert(RBA::Box.new(30, 100, 120, 230))
  cell.shapes(metal_layer).insert(
    RBA::Box.new(gap ? 121 : 120, 100, 210, 230)
  )
  cell.shapes(metal_layer).insert(RBA::Box.new(210, 100, 300, 230))
end

def insert_l_shape(cell, layer)
  points = [
    RBA::Point.new(100, 100),
    RBA::Point.new(230, 100),
    RBA::Point.new(230, 160),
    RBA::Point.new(160, 160),
    RBA::Point.new(160, 230),
    RBA::Point.new(100, 230)
  ]
  cell.shapes(layer).insert(RBA::Polygon.new(points))
end

output_expanded = File.expand_path(required_rd("output", $output))
output_parent = File.realpath(File.dirname(output_expanded))
output_path = File.join(output_parent, File.basename(output_expanded))
fixture_error("refusing to overwrite output: #{output_path}") if
  File.exist?(output_path) || File.symlink?(output_path)

layout = RBA::Layout.new
layout.dbu = 0.0005
metal_layer = layout.layer(101, 0)
via_layer = layout.layer(102, 0)

# Exact 35 nm threshold around the two X-facing sides.
margin_case(
  layout, metal_layer, via_layer, "M2VIA2_X_69", 69, 69, 20, 20
)
margin_case(
  layout, metal_layer, via_layer, "M2VIA2_X_70", 70, 70, 20, 20
)
margin_case(
  layout, metal_layer, via_layer, "M2VIA2_X_71", 71, 71, 20, 20
)

# The same threshold around the two Y-facing sides.
margin_case(
  layout, metal_layer, via_layer, "M2VIA2_Y_69", 20, 20, 69, 69
)
margin_case(
  layout, metal_layer, via_layer, "M2VIA2_Y_70", 20, 20, 70, 70
)
margin_case(
  layout, metal_layer, via_layer, "M2VIA2_Y_71", 20, 20, 71, 71
)

# One good X side and one good Y side are not an opposite pair.
margin_case(
  layout, metal_layer, via_layer, "M2VIA2_PARTIAL_ADJACENT",
  70, 69, 69, 70
)

# X fails by one DBU while the opposite Y pair is exactly at threshold.
margin_case(
  layout, metal_layer, via_layer, "M2VIA2_Y_CHOICE",
  69, 69, 70, 70
)

# Three stored rectangles tile the exact X query. No individual rectangle is
# a witness; union coverage is required. The companion leaves a one-DBU gap.
tiled = layout.create_cell("M2VIA2_UNION_TILED_X")
insert_tiled_x(tiled, metal_layer)
tiled.shapes(via_layer).insert(RBA::Box.new(100, 100, 230, 230))

tiled_gap = layout.create_cell("M2VIA2_UNION_TILED_X_GAP")
insert_tiled_x(tiled_gap, metal_layer, true)
tiled_gap.shapes(via_layer).insert(RBA::Box.new(100, 100, 230, 230))

# The via itself is covered and the left/top projection strips are covered,
# but the right/bottom strips are absent. Adjacent good sides do not qualify.
corner_only = layout.create_cell("M2VIA2_CORNER_ONLY")
corner_only.shapes(metal_layer).insert(RBA::Box.new(30, 100, 230, 230))
corner_only.shapes(metal_layer).insert(RBA::Box.new(100, 100, 230, 300))
corner_only.shapes(via_layer).insert(RBA::Box.new(100, 100, 230, 230))

# Exercise all eight unit orthogonal transforms. Rotations swap the X and Y
# certificate, while mirrors must preserve exact coverage.
hierarchy_leaf = layout.create_cell("M2VIA2_HIERARCHY_LEAF")
insert_tiled_x(hierarchy_leaf, metal_layer)
hierarchy_leaf.shapes(via_layer).insert(RBA::Box.new(100, 100, 230, 230))
hierarchy = layout.create_cell("M2VIA2_HIERARCHY_8")
transform_codes = [
  RBA::Trans::R0, RBA::Trans::R90, RBA::Trans::R180, RBA::Trans::R270,
  RBA::Trans::M0, RBA::Trans::M45, RBA::Trans::M90, RBA::Trans::M135
]
transform_codes.each_with_index do |code, index|
  hierarchy.insert(
    RBA::CellInstArray.new(
      hierarchy_leaf.cell_index,
      RBA::Trans.new(code, 5_000 + index * 2_000, 5_000)
    )
  )
end

# KACTSCN1 accepts the simple Manhattan contour, but the certificate supports
# only rectangular VIA2 operands and must fail closed before launching CUDA.
nonrect = layout.create_cell("M2VIA2_NONRECT_VIA")
nonrect.shapes(metal_layer).insert(RBA::Box.new(0, 0, 500, 500))
insert_l_shape(nonrect, via_layer)

expected_tops = %w[
  M2VIA2_X_69
  M2VIA2_X_70
  M2VIA2_X_71
  M2VIA2_Y_69
  M2VIA2_Y_70
  M2VIA2_Y_71
  M2VIA2_PARTIAL_ADJACENT
  M2VIA2_Y_CHOICE
  M2VIA2_UNION_TILED_X
  M2VIA2_UNION_TILED_X_GAP
  M2VIA2_CORNER_ONLY
  M2VIA2_HIERARCHY_8
  M2VIA2_NONRECT_VIA
].sort
actual_tops = layout.top_cells.map(&:name).sort
fixture_error("unexpected top-cell set: #{actual_tops.join(',')}") unless
  actual_tops == expected_tops

layout.write(output_path)
puts(
  "M2_VIA2_CERTIFICATE_FIXTURE ok path=#{output_path} " \
  "tops=#{expected_tops.length} hierarchy_transforms=8"
)
