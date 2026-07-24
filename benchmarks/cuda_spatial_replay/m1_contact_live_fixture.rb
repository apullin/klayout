# frozen_string_literal: true

# Deterministic live CONTACT/METAL1.3 fixtures for the CUDA empty certificate.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005
contact = layout.layer(10, 0)
metal1 = layout.layer(11, 0)

def contact_cell(layout, name, contact, metal1, metal_boxes, contacts)
  cell = layout.create_cell(name)
  metal_boxes.each { |box| cell.shapes(metal1).insert(box) }
  contacts.each { |shape| cell.shapes(contact).insert(shape) }
  cell
end

contact_cell(
  layout,
  "M1_CONTACT_CLEAN",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 700, 500)],
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(400, 100, 530, 230)]
)

# Projection enclosure accepts equality.  X has exactly 70 DBU on both sides;
# Y is deliberately deficient on both sides.
contact_cell(
  layout,
  "M1_CONTACT_ENCLOSURE_70",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 270, 170)],
  [RBA::Box.new(70, 20, 200, 150)]
)

# One deficient X side remains legal because both Y sides satisfy the rule.
contact_cell(
  layout,
  "M1_CONTACT_ONE_DEFICIENT_SIDE",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 400, 330)],
  [RBA::Box.new(69, 100, 199, 230)]
)

# Deficient opposite X sides remain legal because the perpendicular Y pair
# satisfies the two-opposite-sides projection rule.
contact_cell(
  layout,
  "M1_CONTACT_OPPOSITE_DEFICIENT",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 268, 330)],
  [RBA::Box.new(69, 100, 199, 230)]
)

# Deficient adjacent left/bottom sides leave no complete opposite pair.
contact_cell(
  layout,
  "M1_CONTACT_ADJACENT_DEFICIENT",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 400, 400)],
  [RBA::Box.new(69, 69, 199, 199)]
)

# The historical METAL1.3 expression does not emit for an uncontained contact;
# CONTACT.3 owns that separate violation.  The accelerator nevertheless must
# decline, because absence of an enclosing M1 candidate is not an enclosure
# certificate.
contact_cell(
  layout,
  "M1_CONTACT_OUTSIDE_M1",
  contact,
  metal1,
  [RBA::Box.new(100, 100, 500, 500)],
  [RBA::Box.new(50, 200, 180, 330)]
)

# The following clean CPU cases deliberately leave the qualified accelerator
# domain.  They must conservatively take the historical CPU path.
contact_cell(
  layout,
  "M1_CONTACT_BAD_SIZE",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 700, 500)],
  [RBA::Box.new(100, 100, 229, 230)]
)

contact_cell(
  layout,
  "M1_CONTACT_SPACING_149",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 700, 500)],
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(379, 100, 509, 230)]
)

# CONTACT.2 accepts equality at an axial gap of 150 DBU.
contact_cell(
  layout,
  "M1_CONTACT_SPACING_150",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 700, 500)],
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(380, 100, 510, 230)]
)

contact_cell(
  layout,
  "M1_CONTACT_OVERLAP",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 700, 500)],
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(220, 100, 350, 230)]
)

contact_cell(
  layout,
  "M1_CONTACT_TOUCH",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 700, 500)],
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(230, 100, 360, 230)]
)

# Exact duplicate boxes do not change the merged CONTACT geometry.
contact_cell(
  layout,
  "M1_CONTACT_DUPLICATE",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 500, 400)],
  [RBA::Box.new(100, 100, 230, 230), RBA::Box.new(100, 100, 230, 230)]
)

nonrect = layout.create_cell("M1_CONTACT_NONRECT")
nonrect.shapes(metal1).insert(RBA::Box.new(0, 0, 700, 700))
nonrect.shapes(contact).insert(
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

# The M1 union cleanly encloses the contact, although no individual raw M1
# rectangle does.  This exercises the exact device-side union-strip proof.
contact_cell(
  layout,
  "M1_CONTACT_SPLIT_M1",
  contact,
  metal1,
  [RBA::Box.new(0, 0, 200, 500), RBA::Box.new(200, 0, 500, 500)],
  [RBA::Box.new(185, 150, 315, 280)]
)

leaf = layout.create_cell("M1_CONTACT_HIERARCHY_LEAF")
leaf.shapes(metal1).insert(RBA::Box.new(0, 0, 500, 400))
leaf.shapes(contact).insert(RBA::Box.new(100, 100, 230, 230))
array = layout.create_cell("M1_CONTACT_HIERARCHY_ARRAY")
array.insert(
  RBA::CellInstArray.new(
    leaf.cell_index,
    RBA::Trans.new,
    RBA::Vector.new(800, 0),
    RBA::Vector.new(0, 700),
    2,
    2
  )
)
hierarchy = layout.create_cell("M1_CONTACT_HIERARCHY")
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
  M1_CONTACT_ADJACENT_DEFICIENT
  M1_CONTACT_BAD_SIZE
  M1_CONTACT_CLEAN
  M1_CONTACT_DUPLICATE
  M1_CONTACT_ENCLOSURE_70
  M1_CONTACT_HIERARCHY
  M1_CONTACT_NONRECT
  M1_CONTACT_ONE_DEFICIENT_SIDE
  M1_CONTACT_OPPOSITE_DEFICIENT
  M1_CONTACT_OUTSIDE_M1
  M1_CONTACT_OVERLAP
  M1_CONTACT_SPACING_149
  M1_CONTACT_SPACING_150
  M1_CONTACT_SPLIT_M1
  M1_CONTACT_TOUCH
].sort
actual_tops = layout.top_cells.map(&:name).sort
raise("unexpected top cells: #{actual_tops.join(',')}") unless actual_tops == expected_tops

layout.write(output)
puts("M1_CONTACT_LIVE_FIXTURE ok path=#{output} tops=#{expected_tops.length}")
