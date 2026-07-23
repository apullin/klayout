# frozen_string_literal: true

# Deterministic live M1/VIA1/M2 fixtures for the atomic CUDA transaction.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005
metal1 = layout.layer(11, 0)
via1 = layout.layer(12, 0)
metal2 = layout.layer(13, 0)

def stack_cell(layout, name, metal1, via1, metal2, m1_box, cuts, m2_box)
  cell = layout.create_cell(name)
  cell.shapes(metal1).insert(m1_box)
  cuts.each { |cut| cell.shapes(via1).insert(cut) }
  cell.shapes(metal2).insert(m2_box)
  cell
end

def insert_l_shape(cell, layer)
  cell.shapes(layer).insert(
    RBA::Polygon.new(
      [
        RBA::Point.new(0, 0),
        RBA::Point.new(400, 0),
        RBA::Point.new(400, 160),
        RBA::Point.new(160, 160),
        RBA::Point.new(160, 400),
        RBA::Point.new(0, 400)
      ]
    )
  )
end

def insert_plus_shape(cell, layer)
  cell.shapes(layer).insert(
    RBA::Polygon.new(
      [
        RBA::Point.new(150, 0),
        RBA::Point.new(350, 0),
        RBA::Point.new(350, 160),
        RBA::Point.new(500, 160),
        RBA::Point.new(500, 340),
        RBA::Point.new(350, 340),
        RBA::Point.new(350, 500),
        RBA::Point.new(150, 500),
        RBA::Point.new(150, 340),
        RBA::Point.new(0, 340),
        RBA::Point.new(0, 160),
        RBA::Point.new(150, 160)
      ]
    )
  )
end

stack_cell(
  layout,
  "VIA1_STACK_CLEAN",
  metal1,
  via1,
  metal2,
  RBA::Box.new(0, 0, 700, 500),
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(400, 100, 530, 230)],
  RBA::Box.new(0, 0, 700, 500)
)

# Projection enclosure accepts equality. The cut has exactly 70 DBU on both
# X sides and only 20 DBU on its Y sides in both enclosing metals.
stack_cell(
  layout,
  "VIA1_STACK_ENCLOSURE_70",
  metal1,
  via1,
  metal2,
  RBA::Box.new(0, 0, 270, 170),
  [RBA::Box.new(70, 20, 200, 150)],
  RBA::Box.new(0, 0, 270, 170)
)

stack_cell(
  layout,
  "VIA1_STACK_M1_MISS",
  metal1,
  via1,
  metal2,
  RBA::Box.new(0, 0, 250, 250),
  [RBA::Box.new(60, 60, 190, 190)],
  RBA::Box.new(-100, -100, 400, 400)
)

stack_cell(
  layout,
  "VIA1_STACK_M2_MISS",
  metal1,
  via1,
  metal2,
  RBA::Box.new(-100, -100, 400, 400),
  [RBA::Box.new(60, 60, 190, 190)],
  RBA::Box.new(0, 0, 250, 250)
)

stack_cell(
  layout,
  "VIA1_STACK_OUTSIDE_M1",
  metal1,
  via1,
  metal2,
  RBA::Box.new(100, 100, 500, 500),
  [RBA::Box.new(50, 200, 180, 330)],
  RBA::Box.new(-100, 0, 600, 600)
)

stack_cell(
  layout,
  "VIA1_STACK_OUTSIDE_M2",
  metal1,
  via1,
  metal2,
  RBA::Box.new(-100, 0, 600, 600),
  [RBA::Box.new(50, 200, 180, 330)],
  RBA::Box.new(100, 100, 500, 500)
)

stack_cell(
  layout,
  "VIA1_STACK_BAD_SIZE",
  metal1,
  via1,
  metal2,
  RBA::Box.new(0, 0, 700, 500),
  [RBA::Box.new(100, 100, 229, 230)],
  RBA::Box.new(0, 0, 700, 500)
)

stack_cell(
  layout,
  "VIA1_STACK_SPACING_149",
  metal1,
  via1,
  metal2,
  RBA::Box.new(0, 0, 700, 500),
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(379, 100, 509, 230)],
  RBA::Box.new(0, 0, 700, 500)
)

# The strict spacing rule accepts equality at an axial gap of 150 DBU.
stack_cell(
  layout,
  "VIA1_STACK_SPACING_150",
  metal1,
  via1,
  metal2,
  RBA::Box.new(0, 0, 700, 500),
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(380, 100, 510, 230)],
  RBA::Box.new(0, 0, 700, 500)
)

# Squared integer distance distinguishes the exact 3-4-5 boundary from the
# adjacent violation without floating-point tolerance.
stack_cell(
  layout,
  "VIA1_STACK_DIAGONAL_90_119",
  metal1,
  via1,
  metal2,
  RBA::Box.new(0, 0, 600, 600),
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(320, 349, 450, 479)],
  RBA::Box.new(0, 0, 600, 600)
)

stack_cell(
  layout,
  "VIA1_STACK_DIAGONAL_90_120",
  metal1,
  via1,
  metal2,
  RBA::Box.new(0, 0, 600, 600),
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(320, 350, 450, 480)],
  RBA::Box.new(0, 0, 600, 600)
)

# Exact coincident duplicates leave the merged VIA1 geometry unchanged and
# must be accepted without weakening rejection of non-identical touching cuts.
stack_cell(
  layout,
  "VIA1_STACK_DUPLICATE",
  metal1,
  via1,
  metal2,
  RBA::Box.new(0, 0, 500, 400),
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(100, 100, 230, 230)],
  RBA::Box.new(0, 0, 500, 400)
)

stack_cell(
  layout,
  "VIA1_STACK_TOUCH",
  metal1,
  via1,
  metal2,
  RBA::Box.new(0, 0, 700, 500),
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(230, 100, 360, 230)],
  RBA::Box.new(0, 0, 700, 500)
)

nonrect = layout.create_cell("VIA1_STACK_NONRECT_CUT")
nonrect.shapes(metal1).insert(RBA::Box.new(0, 0, 700, 700))
nonrect.shapes(via1).insert(
  RBA::Polygon.new(
    [
      RBA::Point.new(100, 100),
      RBA::Point.new(300, 100),
      RBA::Point.new(300, 180),
      RBA::Point.new(180, 180),
      RBA::Point.new(180, 300),
      RBA::Point.new(100, 300)
    ]
  )
)
nonrect.shapes(metal2).insert(RBA::Box.new(0, 0, 700, 700))

# The L witness is certified by a horizontal/Y-slab rectangle.
l_slab = layout.create_cell("VIA1_STACK_L_SLAB")
insert_l_shape(l_slab, metal1)
l_slab.shapes(via1).insert(RBA::Box.new(70, 15, 200, 145))
l_slab.shapes(metal2).insert(RBA::Box.new(0, 0, 500, 300))

# This cut straddles the plus polygon's Y=160 boundary. No Y-slab rectangle
# contains it; only the central X slab [150,0;350,500] certifies its Y margins.
plus_slab = layout.create_cell("VIA1_STACK_PLUS_SLAB")
plus_slab.shapes(metal1).insert(RBA::Box.new(0, 0, 500, 500))
plus_slab.shapes(via1).insert(RBA::Box.new(185, 100, 315, 230))
insert_plus_shape(plus_slab, metal2)

leaf = layout.create_cell("VIA1_STACK_HIERARCHY_LEAF")
leaf.shapes(metal1).insert(RBA::Box.new(0, 0, 400, 300))
leaf.shapes(via1).insert(RBA::Box.new(100, 80, 230, 210))
leaf.shapes(metal2).insert(RBA::Box.new(0, 0, 400, 300))
array = layout.create_cell("VIA1_STACK_HIERARCHY_ARRAY")
array.insert(
  RBA::CellInstArray.new(
    leaf.cell_index,
    RBA::Trans.new,
    RBA::Vector.new(700, 0),
    RBA::Vector.new(0, 600),
    2,
    2
  )
)
hierarchy = layout.create_cell("VIA1_STACK_HIERARCHY")
hierarchy.insert(
  RBA::CellInstArray.new(
    array.cell_index,
    RBA::Trans.new(RBA::Trans::R90, 5_000, 5_000)
  )
)
hierarchy.insert(
  RBA::CellInstArray.new(
    array.cell_index,
    RBA::Trans.new(RBA::Trans::M90, 15_000, 5_000)
  )
)

# Leaf/array are implementation cells, not independent test tops.
expected_tops = %w[
  VIA1_STACK_BAD_SIZE
  VIA1_STACK_CLEAN
  VIA1_STACK_DIAGONAL_90_119
  VIA1_STACK_DIAGONAL_90_120
  VIA1_STACK_DUPLICATE
  VIA1_STACK_ENCLOSURE_70
  VIA1_STACK_HIERARCHY
  VIA1_STACK_L_SLAB
  VIA1_STACK_M1_MISS
  VIA1_STACK_M2_MISS
  VIA1_STACK_NONRECT_CUT
  VIA1_STACK_OUTSIDE_M1
  VIA1_STACK_OUTSIDE_M2
  VIA1_STACK_PLUS_SLAB
  VIA1_STACK_SPACING_149
  VIA1_STACK_SPACING_150
  VIA1_STACK_TOUCH
].sort
actual_tops = layout.top_cells.map(&:name).sort
raise("unexpected top cells: #{actual_tops.join(',')}") unless actual_tops == expected_tops

layout.write(output)
puts("VIA1_STACK_LIVE_FIXTURE ok path=#{output} tops=#{expected_tops.length}")
