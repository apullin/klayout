# frozen_string_literal: true

# Deterministic fixtures for the live CONTACT.4 CUDA empty-certificate gate.
#
# At the default 0.5-nm DBU, the FreePDK45 5-nm enclosure limit is exactly
# 10 DBU.  The fixtures deliberately exercise strict/equality boundaries,
# Euclidean endpoint distances, partial projections, collinear degeneracies,
# merged-vs-raw ACTIVE geometry, and hierarchy transforms.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

dbu = ($dbu || "0.0005").to_f
raise("invalid DBU #{dbu}") unless dbu.positive?

layout = RBA::Layout.new
layout.dbu = dbu
active = layout.layer(1, 0)
contact = layout.layer(10, 0)

def contact4_cell(layout, name, active, contact, active_shapes, contact_shapes)
  cell = layout.create_cell(name)
  active_shapes.each { |shape| cell.shapes(active).insert(shape) }
  contact_shapes.each { |shape| cell.shapes(contact).insert(shape) }
  cell
end

# Ordinary clean enclosure.
contact4_cell(
  layout,
  "CONTACT4_CLEAN",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(20, 20, 80, 80)]
)

# The 5-nm check is strict: 9 DBU violates and 10 DBU is accepted.
contact4_cell(
  layout,
  "CONTACT4_GAP_9",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(9, 20, 80, 80)]
)
contact4_cell(
  layout,
  "CONTACT4_GAP_10",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(10, 20, 80, 80)]
)

# Opposite vertical edges have disjoint projections.  Their nearest endpoints
# differ by (6,7) or (6,8): sqrt(85) is below 10, while sqrt(100) is equality.
contact4_cell(
  layout,
  "CONTACT4_ENDPOINT_6_7",
  active,
  contact,
  [RBA::Box.new(0, -100, 100, 100)],
  [RBA::Box.new(6, 107, 50, 150)]
)
contact4_cell(
  layout,
  "CONTACT4_ENDPOINT_6_8",
  active,
  contact,
  [RBA::Box.new(0, -100, 100, 100)],
  [RBA::Box.new(6, 108, 50, 150)]
)

# Only a strict subsegment of the two vertical edges projects onto the other
# edge.  The axial 9/10-DBU pair checks partial-projection clipping and the
# equality boundary independently of the endpoint-only cases above.
contact4_cell(
  layout,
  "CONTACT4_PARTIAL_PROJECTION_9",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(9, 40, 50, 160)]
)
contact4_cell(
  layout,
  "CONTACT4_PARTIAL_PROJECTION_10",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(10, 40, 50, 160)]
)

# The facing vertical edges are collinear.  These cover point touch, a proper
# overlap interval, and a separated endpoint pair.  IncludeZeroDistanceWhen-
# Touching makes the first two particularly important matcher/predicate cases.
contact4_cell(
  layout,
  "CONTACT4_COLLINEAR_TOUCH",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(0, 100, 40, 140)]
)
contact4_cell(
  layout,
  "CONTACT4_COLLINEAR_OVERLAP",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(0, 40, 40, 140)]
)
contact4_cell(
  layout,
  "CONTACT4_COLLINEAR_SEPARATION",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(0, 111, 40, 151)]
)

# Exact duplicates and non-identical overlaps disappear or acquire different
# subsegments when ACTIVE is merged.  Qualified lanes use merged semantics;
# raw-primary lanes must never enter the CUDA seam.
contact4_cell(
  layout,
  "CONTACT4_RAW_DUPLICATE_MERGE",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100), RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(10, 20, 80, 80)]
)
contact4_cell(
  layout,
  "CONTACT4_RAW_OVERLAP_MERGE",
  active,
  contact,
  [RBA::Box.new(0, 0, 85, 100), RBA::Box.new(15, 0, 100, 100)],
  [RBA::Box.new(10, 20, 90, 80)]
)

clean_leaf = layout.create_cell("CONTACT4_HIERARCHY_CLEAN_LEAF")
clean_leaf.shapes(active).insert(RBA::Box.new(0, 0, 100, 100))
clean_leaf.shapes(contact).insert(RBA::Box.new(10, 20, 80, 80))
clean_array = layout.create_cell("CONTACT4_HIERARCHY_CLEAN_ARRAY")
clean_array.insert(
  RBA::CellInstArray.new(
    clean_leaf.cell_index,
    RBA::Trans.new,
    RBA::Vector.new(200, 0),
    RBA::Vector.new(0, 200),
    2,
    2
  )
)
hierarchy_clean = layout.create_cell("CONTACT4_HIERARCHY_CLEAN")
hierarchy_clean.insert(
  RBA::CellInstArray.new(
    clean_array.cell_index,
    RBA::Trans.new(RBA::Trans::R90, 5_000, 5_000)
  )
)
hierarchy_clean.insert(
  RBA::CellInstArray.new(
    clean_array.cell_index,
    RBA::Trans.new(RBA::Trans::M90, 15_000, 5_000)
  )
)

hit_leaf = layout.create_cell("CONTACT4_HIERARCHY_HIT_LEAF")
hit_leaf.shapes(active).insert(RBA::Box.new(0, 0, 100, 100))
hit_leaf.shapes(contact).insert(RBA::Box.new(9, 20, 80, 80))
hit_array = layout.create_cell("CONTACT4_HIERARCHY_HIT_ARRAY")
hit_array.insert(
  RBA::CellInstArray.new(
    hit_leaf.cell_index,
    RBA::Trans.new,
    RBA::Vector.new(200, 0),
    RBA::Vector.new(0, 200),
    2,
    2
  )
)
hierarchy_hit = layout.create_cell("CONTACT4_HIERARCHY_HIT")
hierarchy_hit.insert(
  RBA::CellInstArray.new(
    hit_array.cell_index,
    RBA::Trans.new(RBA::Trans::R270, 5_000, 5_000)
  )
)
hierarchy_hit.insert(
  RBA::CellInstArray.new(
    hit_array.cell_index,
    RBA::Trans.new(RBA::Trans::M45, 15_000, 5_000)
  )
)

expected_tops = %w[
  CONTACT4_CLEAN
  CONTACT4_COLLINEAR_OVERLAP
  CONTACT4_COLLINEAR_SEPARATION
  CONTACT4_COLLINEAR_TOUCH
  CONTACT4_ENDPOINT_6_7
  CONTACT4_ENDPOINT_6_8
  CONTACT4_GAP_10
  CONTACT4_GAP_9
  CONTACT4_HIERARCHY_CLEAN
  CONTACT4_HIERARCHY_HIT
  CONTACT4_PARTIAL_PROJECTION_10
  CONTACT4_PARTIAL_PROJECTION_9
  CONTACT4_RAW_DUPLICATE_MERGE
  CONTACT4_RAW_OVERLAP_MERGE
].sort
actual_tops = layout.top_cells.map(&:name).sort
raise("unexpected top cells: #{actual_tops.join(',')}") unless actual_tops == expected_tops

layout.write(output)
puts(
  "CONTACT4_LIVE_FIXTURE ok path=#{output}" \
  " tops=#{expected_tops.length} dbu=#{layout.dbu}" \
  " hierarchy_contexts=11 hierarchy_occurrences=8"
)
