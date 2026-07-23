# frozen_string_literal: true

# Deterministic, bounded fixtures for projection_enclosure_scene_island.
#
# Each named top cell is exported independently through the existing
# active3_packed_scene_export.rb.  In this use of KACTSCN1, layer slot 0
# (101/0) is enclosing metal and layer slot 1 (102/0) is the enclosed cut.

include RBA

def fixture_error(message)
  raise("projection-enclosure fixture: #{message}")
end

def required_rd(name, value)
  fixture_error("missing -rd #{name}=VALUE") if value.nil? || value.to_s.empty?
  value.to_s
end

def insert_l_shape(cell, layer, x, y)
  points = [
    RBA::Point.new(x, y),
    RBA::Point.new(x + 400, y),
    RBA::Point.new(x + 400, y + 160),
    RBA::Point.new(x + 160, y + 160),
    RBA::Point.new(x + 160, y + 400),
    RBA::Point.new(x, y + 400)
  ]
  cell.shapes(layer).insert(RBA::Polygon.new(points))
end

def insert_plus_shape(cell, layer, x, y)
  points = [
    RBA::Point.new(x + 150, y),
    RBA::Point.new(x + 350, y),
    RBA::Point.new(x + 350, y + 160),
    RBA::Point.new(x + 500, y + 160),
    RBA::Point.new(x + 500, y + 340),
    RBA::Point.new(x + 350, y + 340),
    RBA::Point.new(x + 350, y + 500),
    RBA::Point.new(x + 150, y + 500),
    RBA::Point.new(x + 150, y + 340),
    RBA::Point.new(x, y + 340),
    RBA::Point.new(x, y + 160),
    RBA::Point.new(x + 150, y + 160)
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
cut_layer = layout.layer(102, 0)

# Ordinary clean certificate: the X margins are comfortably above 70 DBU.
clean_box = layout.create_cell("PROJECTION_CLEAN_BOX")
clean_box.shapes(metal_layer).insert(RBA::Box.new(0, 0, 400, 300))
clean_box.shapes(cut_layer).insert(RBA::Box.new(100, 50, 230, 180))

# Equality is accepted.  Both X margins are exactly 70 DBU while neither Y
# margin reaches the rule distance.
exact_boundary = layout.create_cell("PROJECTION_EXACT_BOUNDARY")
exact_boundary.shapes(metal_layer).insert(RBA::Box.new(0, 0, 270, 170))
exact_boundary.shapes(cut_layer).insert(RBA::Box.new(70, 20, 200, 150))

# The island emits both horizontal and vertical slab decompositions for this
# metal L (four rectangles total). Its 130x130 cut is certified by the lower
# horizontal rectangle's X margins.
l_shape = layout.create_cell("PROJECTION_L_SHAPE")
insert_l_shape(l_shape, metal_layer, 0, 0)
l_shape.shapes(cut_layer).insert(RBA::Box.new(70, 15, 200, 145))

# This cut straddles the plus's Y=160 slab boundary, so no horizontal
# decomposition rectangle contains it. The central X slab [150,0;350,500]
# supplies the sole clean certificate through its Y margins.
x_slab_witness = layout.create_cell("PROJECTION_X_SLAB_WITNESS")
insert_plus_shape(x_slab_witness, metal_layer, 0, 0)
x_slab_witness.shapes(cut_layer).insert(RBA::Box.new(185, 100, 315, 230))

# The cut is contained, but all four margins are only 60 DBU.  This must
# request pristine fallback rather than issue a false clean certificate.
missing = layout.create_cell("PROJECTION_MISSING_ENCLOSURE")
missing.shapes(metal_layer).insert(RBA::Box.new(0, 0, 250, 250))
missing.shapes(cut_layer).insert(RBA::Box.new(60, 60, 190, 190))

# Exact coincident cut duplicates do not change the geometric union and are
# accepted explicitly. Both stored cut records must survive the GDS round trip.
duplicate_cuts = layout.create_cell("PROJECTION_DUPLICATE_CUTS")
duplicate_cuts.shapes(metal_layer).insert(RBA::Box.new(0, 0, 700, 400))
duplicate_cuts.shapes(cut_layer).insert(RBA::Box.new(100, 100, 230, 230))
duplicate_cuts.shapes(cut_layer).insert(RBA::Box.new(100, 100, 230, 230))

# Non-identical cuts that share an edge are ambiguous merged geometry and must
# request fallback even though each individual cut is cleanly enclosed.
touching_cuts = layout.create_cell("PROJECTION_TOUCHING_CUTS")
touching_cuts.shapes(metal_layer).insert(RBA::Box.new(0, 0, 700, 400))
touching_cuts.shapes(cut_layer).insert(RBA::Box.new(100, 100, 230, 230))
touching_cuts.shapes(cut_layer).insert(RBA::Box.new(230, 100, 360, 230))

# VIA1 spacing is strict at 150 DBU: an axial gap of 149 violates it, while an
# otherwise identical gap of exactly 150 is clean.
spacing_149 = layout.create_cell("PROJECTION_SPACING_149")
spacing_149.shapes(metal_layer).insert(RBA::Box.new(0, 0, 700, 400))
spacing_149.shapes(cut_layer).insert(RBA::Box.new(100, 100, 230, 230))
spacing_149.shapes(cut_layer).insert(RBA::Box.new(379, 100, 509, 230))

spacing_150 = layout.create_cell("PROJECTION_SPACING_150")
spacing_150.shapes(metal_layer).insert(RBA::Box.new(0, 0, 700, 400))
spacing_150.shapes(cut_layer).insert(RBA::Box.new(100, 100, 230, 230))
spacing_150.shapes(cut_layer).insert(RBA::Box.new(380, 100, 510, 230))

# Diagonal spacing uses exact squared integer distance. A (90,120) gap is
# exactly 150 DBU and passes; changing only the Y gap to 119 must violate.
diagonal_90_120 = layout.create_cell("PROJECTION_DIAGONAL_90_120")
diagonal_90_120.shapes(metal_layer).insert(RBA::Box.new(0, 0, 600, 600))
diagonal_90_120.shapes(cut_layer).insert(RBA::Box.new(100, 100, 230, 230))
diagonal_90_120.shapes(cut_layer).insert(RBA::Box.new(320, 350, 450, 480))

diagonal_90_119 = layout.create_cell("PROJECTION_DIAGONAL_90_119")
diagonal_90_119.shapes(metal_layer).insert(RBA::Box.new(0, 0, 600, 600))
diagonal_90_119.shapes(cut_layer).insert(RBA::Box.new(100, 100, 230, 230))
diagonal_90_119.shapes(cut_layer).insert(RBA::Box.new(320, 349, 450, 479))

# One clean leaf is expanded through a 2x2 regular array beneath both an R90
# and a mirrored M90 parent.  The expected expansion is:
#   1 root + 2 array contexts + 8 leaf contexts = 11 contexts,
#   8 metal occurrences and 8 cut occurrences.
hierarchy_leaf = layout.create_cell("PROJECTION_HIERARCHY_LEAF")
hierarchy_leaf.shapes(metal_layer).insert(RBA::Box.new(0, 0, 300, 200))
hierarchy_leaf.shapes(cut_layer).insert(RBA::Box.new(70, 35, 200, 165))
hierarchy_array = layout.create_cell("PROJECTION_HIERARCHY_ARRAY")
hierarchy_array.insert(
  RBA::CellInstArray.new(
    hierarchy_leaf.cell_index,
    RBA::Trans.new,
    RBA::Vector.new(500, 0),
    RBA::Vector.new(0, 400),
    2,
    2
  )
)
hierarchy = layout.create_cell("PROJECTION_HIERARCHY")
hierarchy.insert(
  RBA::CellInstArray.new(
    hierarchy_array.cell_index,
    RBA::Trans.new(RBA::Trans::R90, 5_000, 5_000)
  )
)
hierarchy.insert(
  RBA::CellInstArray.new(
    hierarchy_array.cell_index,
    RBA::Trans.new(RBA::Trans::M90, 15_000, 5_000)
  )
)

# KACTSCN1 accepts this simple Manhattan cut, but the bounded projection
# certificate deliberately supports rectangle cuts only and must fail closed.
nonrect_cut = layout.create_cell("PROJECTION_NONRECT_CUT")
nonrect_cut.shapes(metal_layer).insert(RBA::Box.new(0, 0, 700, 700))
insert_l_shape(nonrect_cut, cut_layer, 100, 100)

expected_tops = %w[
  PROJECTION_CLEAN_BOX
  PROJECTION_EXACT_BOUNDARY
  PROJECTION_L_SHAPE
  PROJECTION_X_SLAB_WITNESS
  PROJECTION_MISSING_ENCLOSURE
  PROJECTION_DUPLICATE_CUTS
  PROJECTION_TOUCHING_CUTS
  PROJECTION_SPACING_149
  PROJECTION_SPACING_150
  PROJECTION_DIAGONAL_90_120
  PROJECTION_DIAGONAL_90_119
  PROJECTION_HIERARCHY
  PROJECTION_NONRECT_CUT
].sort
actual_tops = layout.top_cells.map(&:name).sort
fixture_error("unexpected top-cell set: #{actual_tops.join(',')}") unless
  actual_tops == expected_tops

layout.write(output_path)
puts(
  "PROJECTION_ENCLOSURE_SCENE_FIXTURE ok path=#{output_path} " \
  "tops=#{expected_tops.length} hierarchy_contexts=11 " \
  "hierarchy_metals=8 hierarchy_cuts=8"
)
