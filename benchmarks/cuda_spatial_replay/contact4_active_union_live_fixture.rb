# frozen_string_literal: true

# Deterministic fixtures for the fused raw-ACTIVE-union/CONTACT.4 live gate.
#
# At the qualified 0.5-nm DBU, the FreePDK45 5-nm enclosure limit is exactly
# 10 DBU.  In addition to the strict 9/10 boundary, two cases contain raw
# ACTIVE edges less than 10 DBU from CONTACT which disappear under exact set
# union.  A conservative raw-edge scan would decline those cases; the fused
# implementation must certify their exact merged boundary instead.

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
  "CONTACT4_ACTIVE_UNION_CLEAN",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(20, 20, 80, 80)]
)

# The rule is strict: 9 DBU violates and equality at 10 DBU is accepted.
contact4_cell(
  layout,
  "CONTACT4_ACTIVE_UNION_GAP_9",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(9, 20, 80, 80)]
)
contact4_cell(
  layout,
  "CONTACT4_ACTIVE_UNION_GAP_10",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(10, 20, 80, 80)]
)

# The two ACTIVE boxes union to (0,0)-(100,100).  Their raw x=40 and x=60
# edges are each only 5 DBU from CONTACT, but neither edge survives union.
contact4_cell(
  layout,
  "CONTACT4_ACTIVE_UNION_OVERLAP_CLEAN",
  active,
  contact,
  [RBA::Box.new(0, 0, 60, 100), RBA::Box.new(40, 0, 100, 100)],
  [RBA::Box.new(45, 20, 55, 80)]
)

# Exact duplicate inner shapes exercise multiplicity as well as containment.
# All inner-shape edges disappear into the outer ACTIVE box.  The x=35
# and x=65 raw edges are only 5 DBU from CONTACT, while the exact union has a
# clean 40-DBU enclosure.
contact4_cell(
  layout,
  "CONTACT4_ACTIVE_UNION_DUPLICATE_CLEAN",
  active,
  contact,
  [
    RBA::Box.new(0, 0, 100, 100),
    RBA::Box.new(35, 10, 65, 90),
    RBA::Box.new(35, 10, 65, 90)
  ],
  [RBA::Box.new(40, 20, 60, 80)]
)

# ACTIVE and CONTACT live in distinct siblings and are instantiated through
# every orthogonal unit transform.  This catches context-composition and
# reflected-boundary direction errors that a co-located leaf would hide.
hier_active_leaf =
  layout.create_cell("CONTACT4_ACTIVE_UNION_HIER_ACTIVE_LEAF")
hier_active_leaf.shapes(active).insert(RBA::Box.new(0, 0, 100, 100))

hier_contact_clean_leaf =
  layout.create_cell("CONTACT4_ACTIVE_UNION_HIER_CONTACT_CLEAN_LEAF")
hier_contact_clean_leaf.shapes(contact).insert(RBA::Box.new(20, 20, 80, 80))

hier_contact_hit_leaf =
  layout.create_cell("CONTACT4_ACTIVE_UNION_HIER_CONTACT_HIT_LEAF")
hier_contact_hit_leaf.shapes(contact).insert(RBA::Box.new(9, 20, 80, 80))

hierarchy_clean =
  layout.create_cell("CONTACT4_ACTIVE_UNION_HIERARCHY_CLEAN")
hierarchy_hit =
  layout.create_cell("CONTACT4_ACTIVE_UNION_HIERARCHY_HIT")

transforms = [
  RBA::Trans::R0, RBA::Trans::R90, RBA::Trans::R180, RBA::Trans::R270,
  RBA::Trans::M0, RBA::Trans::M45, RBA::Trans::M90, RBA::Trans::M135
]
transforms.each_with_index do |transform, index|
  dx = 10_000 + index * 1_000
  dy = 10_000
  [hierarchy_clean, hierarchy_hit].each do |top|
    top.insert(
      RBA::CellInstArray.new(
        hier_active_leaf.cell_index,
        RBA::Trans.new(transform, dx, dy)
      )
    )
  end
  hierarchy_clean.insert(
    RBA::CellInstArray.new(
      hier_contact_clean_leaf.cell_index,
      RBA::Trans.new(transform, dx, dy)
    )
  )
  hierarchy_hit.insert(
    RBA::CellInstArray.new(
      hier_contact_hit_leaf.cell_index,
      RBA::Trans.new(transform, dx, dy)
    )
  )
end

# Anchor the exact live-serializer qualifications in the generated artifact.
layout.each_cell do |cell|
  { "ACTIVE" => active, "CONTACT" => contact }.each do |name, layer|
    cell.shapes(layer).each do |shape|
      raise("#{cell.name}/#{name}: shape has properties") unless shape.prop_id == 0
      raise("#{cell.name}/#{name}: shape is not polygonal") unless
        shape.is_box? || shape.is_polygon?
      polygon = shape.polygon
      raise("#{cell.name}/#{name}: polygon has holes") unless polygon.holes == 0
      twice_area = 0
      edge_count = 0
      polygon.each_edge do |edge|
        raise("#{cell.name}/#{name}: edge is not Manhattan") unless
          edge.p1.x == edge.p2.x || edge.p1.y == edge.p2.y
        twice_area += edge.p1.x * edge.p2.y - edge.p2.x * edge.p1.y
        edge_count += 1
      end
      raise("#{cell.name}/#{name}: polygon is not a clockwise rectangle") unless
        edge_count == 4 && twice_area.negative?
    end
  end
end

expected_tops = %w[
  CONTACT4_ACTIVE_UNION_CLEAN
  CONTACT4_ACTIVE_UNION_DUPLICATE_CLEAN
  CONTACT4_ACTIVE_UNION_GAP_10
  CONTACT4_ACTIVE_UNION_GAP_9
  CONTACT4_ACTIVE_UNION_HIERARCHY_CLEAN
  CONTACT4_ACTIVE_UNION_HIERARCHY_HIT
  CONTACT4_ACTIVE_UNION_OVERLAP_CLEAN
].sort
actual_tops = layout.top_cells.map(&:name).sort
raise("unexpected top cells: #{actual_tops.join(',')}") unless actual_tops == expected_tops

layout.write(output)
puts(
  "CONTACT4_ACTIVE_UNION_LIVE_FIXTURE ok path=#{output}" \
  " tops=#{expected_tops.length} dbu=#{layout.dbu}" \
  " hierarchy_transforms=#{transforms.length}" \
  " hierarchy_leaf_cells=3 duplicate_active_shapes=2"
)
