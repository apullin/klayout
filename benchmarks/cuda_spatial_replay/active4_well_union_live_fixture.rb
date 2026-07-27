# frozen_string_literal: true

# Deterministic exact-set fixtures for the resident ACTIVE subset-of-WELL
# certificate.  The layouts deliberately cover cases that a boundary-only
# test cannot prove, including a far-away ACTIVE polygon and a WELL hole.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005
active = layout.layer(1, 0)
pwell = layout.layer(2, 0)
nwell = layout.layer(3, 0)

def box(cell, layer, left, bottom, right, top)
  cell.shapes(layer).insert(RBA::Box.new(left, bottom, right, top))
end

inside = layout.create_cell("ACTIVE4_INSIDE")
box(inside, nwell, 0, 0, 1000, 1000)
box(inside, pwell, 2000, 0, 3000, 1000)
box(inside, active, 200, 200, 800, 800)

# Set containment includes coincident boundaries.
boundary_touch = layout.create_cell("ACTIVE4_BOUNDARY_TOUCH")
box(boundary_touch, nwell, 0, 0, 1000, 1000)
box(boundary_touch, pwell, 2000, 0, 3000, 1000)
box(boundary_touch, active, 0, 200, 800, 800)

# The ACTIVE operands overlap; the exact set is still contained.
overlap_active = layout.create_cell("ACTIVE4_OVERLAP_ACTIVE")
box(overlap_active, nwell, 0, 0, 1000, 1000)
box(overlap_active, pwell, 2000, 0, 3000, 1000)
box(overlap_active, active, 100, 100, 650, 650)
box(overlap_active, active, 350, 350, 900, 900)

# An internal raw NWELL/PWELL boundary crosses ACTIVE, but disappears in the
# exact union.  Boundary-pair shortcuts can false-positive here.
internal_boundary = layout.create_cell("ACTIVE4_INTERNAL_WELL_BOUNDARY")
box(internal_boundary, nwell, 0, 0, 650, 1000)
box(internal_boundary, pwell, 350, 0, 1000, 1000)
box(internal_boundary, active, 450, 200, 550, 800)

partial_outside = layout.create_cell("ACTIVE4_PARTIAL_OUTSIDE")
box(partial_outside, nwell, 0, 0, 1000, 1000)
box(partial_outside, pwell, 2000, 0, 3000, 1000)
box(partial_outside, active, -50, 200, 500, 800)

# There are no nearby WELL edges or slabs.  A candidate-pair-only predicate
# would incorrectly call this clean.
far_outside = layout.create_cell("ACTIVE4_FAR_OUTSIDE")
box(far_outside, nwell, 0, 0, 1000, 1000)
box(far_outside, pwell, 2000, 0, 3000, 1000)
box(far_outside, active, 5000, 200, 5500, 800)

# Four rectangles form a WELL ring with a true uncovered hole.
well_hole = layout.create_cell("ACTIVE4_WELL_HOLE")
box(well_hole, nwell, 0, 0, 1000, 400)
box(well_hole, nwell, 0, 600, 1000, 1000)
box(well_hole, nwell, 0, 400, 400, 600)
box(well_hole, nwell, 600, 400, 1000, 600)
box(well_hole, pwell, 2000, 0, 3000, 1000)
box(well_hole, active, 425, 425, 575, 575)

# One shared hierarchy is instantiated under every orthogonal rotation and
# reflection.  ACTIVE crosses the raw well boundary but is inside their union.
hierarchy_leaf = layout.create_cell("ACTIVE4_HIERARCHY_LEAF")
box(hierarchy_leaf, nwell, 0, 0, 650, 1000)
box(hierarchy_leaf, pwell, 350, 0, 1000, 1000)
box(hierarchy_leaf, active, 200, 200, 800, 800)
hierarchy = layout.create_cell("ACTIVE4_HIERARCHY_TRANSFORMS")
[
  RBA::Trans::R0, RBA::Trans::R90, RBA::Trans::R180, RBA::Trans::R270,
  RBA::Trans::M0, RBA::Trans::M45, RBA::Trans::M90, RBA::Trans::M135
].each_with_index do |transform, index|
  hierarchy.insert(
    RBA::CellInstArray.new(
      hierarchy_leaf.cell_index,
      RBA::Trans.new(transform, 5000 + index * 3000, 5000)
    )
  )
end

expected_tops = %w[
  ACTIVE4_BOUNDARY_TOUCH
  ACTIVE4_FAR_OUTSIDE
  ACTIVE4_HIERARCHY_TRANSFORMS
  ACTIVE4_INSIDE
  ACTIVE4_INTERNAL_WELL_BOUNDARY
  ACTIVE4_OVERLAP_ACTIVE
  ACTIVE4_PARTIAL_OUTSIDE
  ACTIVE4_WELL_HOLE
].sort
actual_tops = layout.top_cells.map(&:name).sort
raise("unexpected top cells: #{actual_tops.join(',')}") unless actual_tops == expected_tops

layout.write(output)
puts("ACTIVE4_WELL_UNION_LIVE_FIXTURE ok path=#{output} tops=8 dbu=#{layout.dbu}")
