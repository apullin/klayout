# frozen_string_literal: true

# Deterministic CPU fixtures for the fail-closed IMPLANT.1/.2 deck rewrite.

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

def insert_gate(cell, active, poly, box)
  cell.shapes(active).insert(box)
  cell.shapes(poly).insert(box)
end

# Both projection gaps are exactly equal to their limits: 140 DBU = 70 nm
# from implant to gate and 50 DBU = 25 nm from implant to contact.
boundary = layout.create_cell("IMPLANT12_BOUNDARY")
boundary.shapes(nplus).insert(RBA::Box.new(0, 0, 400, 400))
insert_gate(
  boundary,
  active,
  poly,
  RBA::Box.new(540, 100, 800, 300)
)
boundary.shapes(cont).insert(RBA::Box.new(450, 100, 580, 230))

# One DBU inside only the 70 nm limit.
implant1_hit = layout.create_cell("IMPLANT12_IMPLANT1_HIT")
implant1_hit.shapes(nplus).insert(RBA::Box.new(0, 0, 400, 400))
insert_gate(
  implant1_hit,
  active,
  poly,
  RBA::Box.new(539, 100, 800, 300)
)

# One DBU inside only the 25 nm limit. Use pplus to cover both contributors to
# the merged implant layer.
implant2_hit = layout.create_cell("IMPLANT12_IMPLANT2_HIT")
implant2_hit.shapes(pplus).insert(RBA::Box.new(0, 0, 400, 400))
implant2_hit.shapes(cont).insert(RBA::Box.new(449, 100, 579, 230))

# Exercise both ordered result categories through a transformed hierarchy.
both_leaf = layout.create_cell("IMPLANT12_BOTH_HIT_LEAF")
both_leaf.shapes(nplus).insert(RBA::Box.new(0, 0, 400, 400))
insert_gate(
  both_leaf,
  active,
  poly,
  RBA::Box.new(539, 100, 800, 300)
)
both_leaf.shapes(cont).insert(RBA::Box.new(449, 100, 579, 230))

both_hit = layout.create_cell("IMPLANT12_BOTH_HIT")
both_hit.insert(
  RBA::CellInstArray.new(
    both_leaf.cell_index,
    RBA::Trans.new(RBA::Trans::M90, 5_000, 7_000)
  )
)

expected_tops = %w[
  IMPLANT12_BOUNDARY
  IMPLANT12_BOTH_HIT
  IMPLANT12_IMPLANT1_HIT
  IMPLANT12_IMPLANT2_HIT
].sort
actual_tops = layout.top_cells.map(&:name).sort
unless actual_tops == expected_tops
  raise("unexpected top cells: #{actual_tops.join(',')}")
end

layout.write(output)
puts("IMPLANT12_LIVE_FIXTURE ok path=#{output} tops=#{expected_tops.length}")
