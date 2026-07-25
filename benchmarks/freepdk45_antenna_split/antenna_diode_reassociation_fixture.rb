# frozen_string_literal: true

# A deterministic hierarchical fixture for proving the FreePDK45 antenna
# diode Boolean reassociation:
#
#   nplus & (active - nwell) == (nplus & active) - nwell
#
# The fixture deliberately distributes operands across hierarchy levels,
# includes partial WELL cuts and empty/disjoint components, and instantiates
# nonempty geometry through all eight orthogonal transforms and an array.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.001

active = layout.layer(1, 0)
nwell = layout.layer(3, 0)
nplus = layout.layer(4, 0)

partial = layout.create_cell("ANTENNA_DIODE_PARTIAL_WELL")
partial.shapes(active).insert(RBA::Box.new(0, 0, 1000, 600))
partial.shapes(nplus).insert(RBA::Box.new(100, 100, 900, 500))
partial.shapes(nwell).insert(RBA::Box.new(350, -50, 650, 650))

empty_nplus = layout.create_cell("ANTENNA_DIODE_EMPTY_NPLUS")
empty_nplus.shapes(active).insert(RBA::Box.new(0, 0, 600, 400))
empty_nplus.shapes(nwell).insert(RBA::Box.new(200, -50, 400, 450))

empty_active = layout.create_cell("ANTENNA_DIODE_EMPTY_ACTIVE")
empty_active.shapes(nplus).insert(RBA::Box.new(0, 0, 600, 400))
empty_active.shapes(nwell).insert(RBA::Box.new(200, -50, 400, 450))

empty_nwell = layout.create_cell("ANTENNA_DIODE_EMPTY_NWELL")
empty_nwell.shapes(active).insert(RBA::Box.new(0, 0, 400, 300))
empty_nwell.shapes(nplus).insert(RBA::Box.new(100, -50, 300, 350))

disjoint = layout.create_cell("ANTENNA_DIODE_DISJOINT")
disjoint.shapes(active).insert(RBA::Box.new(0, 0, 300, 300))
disjoint.shapes(nplus).insert(RBA::Box.new(500, 0, 800, 300))
disjoint.shapes(nwell).insert(RBA::Box.new(100, 100, 200, 200))

active_child = layout.create_cell("ANTENNA_DIODE_ACTIVE_CHILD")
active_child.shapes(active).insert(RBA::Box.new(0, 0, 1000, 500))

nplus_child = layout.create_cell("ANTENNA_DIODE_NPLUS_CHILD")
nplus_child.shapes(nplus).insert(RBA::Box.new(100, 100, 900, 400))

parent_cross = layout.create_cell("ANTENNA_DIODE_PARENT_CHILD_OVERLAP")
parent_cross.insert(
  RBA::CellInstArray.new(active_child.cell_index, RBA::Trans.new)
)
parent_cross.insert(
  RBA::CellInstArray.new(nplus_child.cell_index, RBA::Trans.new)
)
# These parent-level shapes overlap the two child-level operands.  The ACTIVE
# shape also crosses the boundary of the child ACTIVE shape.
parent_cross.shapes(active).insert(RBA::Box.new(800, 0, 1200, 500))
parent_cross.shapes(nwell).insert(RBA::Box.new(400, -50, 600, 550))

top = layout.create_cell("ANTENNA_DIODE_REASSOCIATION")

transforms = [
  RBA::Trans::R0, RBA::Trans::R90, RBA::Trans::R180, RBA::Trans::R270,
  RBA::Trans::M0, RBA::Trans::M45, RBA::Trans::M90, RBA::Trans::M135
]
transforms.each_with_index do |transform, index|
  top.insert(
    RBA::CellInstArray.new(
      partial.cell_index,
      RBA::Trans.new(transform, 5000 + index * 3000, 5000)
    )
  )
end

top.insert(
  RBA::CellInstArray.new(
    empty_nplus.cell_index, RBA::Trans.new(5000, 12_000)
  )
)
top.insert(
  RBA::CellInstArray.new(
    empty_active.cell_index, RBA::Trans.new(9000, 12_000)
  )
)
top.insert(
  RBA::CellInstArray.new(
    empty_nwell.cell_index, RBA::Trans.new(13_000, 12_000)
  )
)
top.insert(
  RBA::CellInstArray.new(
    disjoint.cell_index, RBA::Trans.new(17_000, 12_000)
  )
)

# A 3-by-2 array of a hierarchy whose operands cross parent/child boundaries.
top.insert(
  RBA::CellInstArray.new(
    parent_cross.cell_index,
    RBA::Trans.new(RBA::Trans::R90, 5000, 20_000),
    RBA::Vector.new(3000, 0),
    RBA::Vector.new(0, 3000),
    3,
    2
  )
)

layout.each_cell do |cell|
  [active, nwell, nplus].each do |layer|
    cell.shapes(layer).each do |shape|
      raise("#{cell.name}: fixture shape has properties") unless shape.prop_id == 0
      raise("#{cell.name}: fixture shape is not a box") unless shape.is_box?
    end
  end
end

tops = layout.top_cells.map(&:name)
raise("unexpected top cells: #{tops.join(',')}") unless tops == [top.name]

layout.write(output)
puts(
  "ANTENNA_DIODE_REASSOCIATION_FIXTURE ok path=#{output}" \
  " top=#{top.name} transforms=#{transforms.length}" \
  " parent_child_array=3x2"
)
