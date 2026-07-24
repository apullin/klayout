# frozen_string_literal: true

# A deterministic hierarchical FreePDK45 antenna ladder.  Every M1-M4 target
# has more than 300 times the gate area, with no protecting diode.  This makes
# both sides of the antenna_m1_m2 / antenna_m3_m10 ownership boundary nonempty.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.001

active = layout.layer(1, 0)
poly = layout.layer(9, 0)
cont = layout.layer(10, 0)
metal1 = layout.layer(11, 0)
via1 = layout.layer(12, 0)
metal2 = layout.layer(13, 0)
via2 = layout.layer(14, 0)
metal3 = layout.layer(15, 0)
via3 = layout.layer(16, 0)
metal4 = layout.layer(17, 0)

leaf = layout.create_cell("FREEPDK45_ANTENNA_LEAF")
top = layout.create_cell("FREEPDK45_ANTENNA_SPLIT")

# Gate area is 0.0025 um^2.  Each connected metal rectangle is over 1 um^2,
# so M1-M4 all exceed the deck's 300:1 threshold.
leaf.shapes(active).insert(RBA::Box.new(0, 0, 50, 50))
leaf.shapes(poly).insert(RBA::Box.new(0, 0, 250, 50))
leaf.shapes(cont).insert(RBA::Box.new(185, -5, 250, 60))
leaf.shapes(metal1).insert(RBA::Box.new(185, -100, 1300, 1000))
leaf.shapes(via1).insert(RBA::Box.new(1200, 0, 1265, 65))
leaf.shapes(metal2).insert(RBA::Box.new(1200, -100, 2315, 1000))
leaf.shapes(via2).insert(RBA::Box.new(2215, 0, 2280, 65))
leaf.shapes(metal3).insert(RBA::Box.new(2215, -100, 3330, 1000))
leaf.shapes(via3).insert(RBA::Box.new(3230, 0, 3295, 65))
leaf.shapes(metal4).insert(RBA::Box.new(3230, -100, 4345, 1000))

top.insert(RBA::CellInstArray.new(leaf.cell_index, RBA::Trans.new))
top.insert(
  RBA::CellInstArray.new(
    leaf.cell_index,
    RBA::Trans.new(RBA::Trans::M90, 12_000, 8_000)
  )
)

tops = layout.top_cells.map(&:name)
raise("unexpected top cells: #{tops.join(',')}") unless tops == [top.name]

layout.write(output)
puts("FREEPDK45_ANTENNA_FIXTURE ok path=#{output} top=#{top.name} instances=2")
