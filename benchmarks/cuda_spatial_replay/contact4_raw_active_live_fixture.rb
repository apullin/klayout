# frozen_string_literal: true

# Deterministic fixtures for the early raw-ACTIVE CONTACT.4 live gate.
#
# At the qualified 0.5-nm DBU, the 5-nm CONTACT.4 limit is exactly 10 DBU.
# The false-positive case is deliberately clean after ACTIVE union, but has
# internal raw ACTIVE edges only 5 DBU from CONTACT.  The early conservative
# certificate must therefore report raw hits and fall through to the existing
# merged-ACTIVE certificate.

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

# Ordinary clean enclosure.  The early raw-ACTIVE profile must certify this
# without constructing merged ACTIVE or invoking the established late profile.
contact4_cell(
  layout,
  "CONTACT4_RAW_ACTIVE_CLEAN",
  active,
  contact,
  [
    RBA::Box.new(0, 0, 100, 100),
    RBA::Box.new(90, 0, 140, 100)
  ],
  [RBA::Box.new(20, 20, 70, 80)]
)

# The union is the clean outer box (0,0)-(100,100).  Before union, however,
# x=60 on the first ACTIVE box is 5 DBU from CONTACT's x=55 edge, and x=40 on
# the second ACTIVE box is 5 DBU from CONTACT's x=45 edge.  Those conservative
# raw hits must fall through; they are not publishable CONTACT.4 markers.
contact4_cell(
  layout,
  "CONTACT4_RAW_ACTIVE_FALSE_POSITIVE",
  active,
  contact,
  [RBA::Box.new(0, 0, 60, 100), RBA::Box.new(40, 0, 100, 100)],
  [RBA::Box.new(45, 20, 55, 80)]
)

# A real strict 9-DBU enclosure violation.  Both early raw and late merged
# profiles must decline publication and leave the pristine CPU rule to emit
# the marker.
contact4_cell(
  layout,
  "CONTACT4_RAW_ACTIVE_TRUE_HIT",
  active,
  contact,
  [RBA::Box.new(0, 0, 100, 100)],
  [RBA::Box.new(9, 20, 80, 80)]
)

# Exercise all eight exact orthogonal transforms with ACTIVE and CONTACT in
# different sibling cells.  This prevents a co-located leaf from hiding a
# context-composition or cross-sibling candidate bug.
active_leaf = layout.create_cell("CONTACT4_RAW_ACTIVE_HIER_ACTIVE_LEAF")
active_leaf.shapes(active).insert(RBA::Box.new(0, 0, 100, 100))
contact_leaf = layout.create_cell("CONTACT4_RAW_ACTIVE_HIER_CONTACT_LEAF")
contact_leaf.shapes(contact).insert(RBA::Box.new(20, 20, 80, 80))
hierarchy_clean = layout.create_cell("CONTACT4_RAW_ACTIVE_HIERARCHY_CLEAN")

transforms = [
  RBA::Trans::R0, RBA::Trans::R90, RBA::Trans::R180, RBA::Trans::R270,
  RBA::Trans::M0, RBA::Trans::M45, RBA::Trans::M90, RBA::Trans::M135
]
transforms.each_with_index do |transform, index|
  dx = 10_000 + index * 1_000
  dy = 10_000
  hierarchy_clean.insert(
    RBA::CellInstArray.new(
      active_leaf.cell_index,
      RBA::Trans.new(transform, dx, dy)
    )
  )
  hierarchy_clean.insert(
    RBA::CellInstArray.new(
      contact_leaf.cell_index,
      RBA::Trans.new(transform, dx, dy)
    )
  )
end

# Anchor the geometry qualifications expected by the live serializer.
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
  CONTACT4_RAW_ACTIVE_CLEAN
  CONTACT4_RAW_ACTIVE_FALSE_POSITIVE
  CONTACT4_RAW_ACTIVE_HIERARCHY_CLEAN
  CONTACT4_RAW_ACTIVE_TRUE_HIT
].sort
actual_tops = layout.top_cells.map(&:name).sort
raise("unexpected top cells: #{actual_tops.join(',')}") unless actual_tops == expected_tops

layout.write(output)
puts(
  "CONTACT4_RAW_ACTIVE_LIVE_FIXTURE ok path=#{output}" \
  " tops=#{expected_tops.length} dbu=#{layout.dbu}" \
  " hierarchy_contexts=17 indexed_contact_contexts=8" \
  " streamed_active_contexts=8"
)
