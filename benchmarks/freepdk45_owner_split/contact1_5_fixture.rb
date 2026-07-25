# frozen_string_literal: true

# Deterministic hierarchical nonempty fixture for the FreePDK45
# implant_contact/contact owner boundary.
#
# The qualified FreePDK45 layout uses a 0.5-nm DBU.  Contact width, contact
# spacing, and contact enclosure limits are therefore exactly 130, 150, and
# 10 DBU respectively.  Each focused leaf violates one CONTACT.1-.5 rule
# without relying on a violation from another leaf.  A sixth leaf overlaps
# NPLUS and PPLUS to keep the retained implant owner nonempty.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005

active = layout.layer(1, 0)
nplus = layout.layer(4, 0)
pplus = layout.layer(5, 0)
poly = layout.layer(9, 0)
cont = layout.layer(10, 0)
metal1 = layout.layer(11, 0)

contact1 = layout.create_cell("CONTACT1_BAD_WIDTH")
contact1.shapes(metal1).insert(RBA::Box.new(0, 0, 600, 500))
# 125x130 DBU: two edges differ from the exact 65-nm/130-DBU length.
contact1.shapes(cont).insert(RBA::Box.new(100, 100, 225, 230))

contact2 = layout.create_cell("CONTACT2_BAD_SPACING")
contact2.shapes(metal1).insert(RBA::Box.new(0, 0, 900, 500))
contact2.shapes(cont).insert(RBA::Box.new(100, 100, 230, 230))
# The axial gap is 145 DBU, below the 75-nm/150-DBU limit.
contact2.shapes(cont).insert(RBA::Box.new(375, 100, 505, 230))

contact3 = layout.create_cell("CONTACT3_OUTSIDE_ALLOWED_LAYERS")
contact3.shapes(cont).insert(RBA::Box.new(100, 100, 230, 230))

contact4 = layout.create_cell("CONTACT4_BAD_ACTIVE_ENCLOSURE")
contact4.shapes(active).insert(RBA::Box.new(0, 0, 500, 500))
# The left ACTIVE enclosure is 5 DBU; every other side is comfortably legal.
contact4.shapes(cont).insert(RBA::Box.new(5, 150, 135, 280))

contact5 = layout.create_cell("CONTACT5_BAD_POLY_ENCLOSURE")
contact5.shapes(poly).insert(RBA::Box.new(0, 0, 500, 500))
# The left POLY enclosure is 5 DBU; every other side is comfortably legal.
contact5.shapes(cont).insert(RBA::Box.new(5, 150, 135, 280))

implant = layout.create_cell("IMPLANT5_OVERLAP")
implant.shapes(nplus).insert(RBA::Box.new(0, 0, 300, 300))
implant.shapes(pplus).insert(RBA::Box.new(150, 150, 450, 450))

top = layout.create_cell("FREEPDK45_CONTACT_OWNER_SPLIT")
instances = [
  [contact1, RBA::Trans.new(RBA::Trans::R0, 0, 0)],
  [contact2, RBA::Trans.new(RBA::Trans::R90, 10_000, 0)],
  [contact3, RBA::Trans.new(RBA::Trans::R180, 20_000, 2_000)],
  [contact4, RBA::Trans.new(RBA::Trans::R270, 30_000, 2_000)],
  [contact5, RBA::Trans.new(RBA::Trans::M0, 40_000, 0)],
  [implant, RBA::Trans.new(RBA::Trans::M90, 50_000, 2_000)]
]
instances.each do |cell, transform|
  top.insert(RBA::CellInstArray.new(cell.cell_index, transform))
end

tops = layout.top_cells.map(&:name)
raise("unexpected top cells: #{tops.join(',')}") unless tops == [top.name]

layout.write(output)
puts(
  "FREEPDK45_CONTACT_OWNER_FIXTURE ok path=#{output}" \
  " top=#{top.name} leaves=#{instances.length} dbu=#{layout.dbu}"
)
