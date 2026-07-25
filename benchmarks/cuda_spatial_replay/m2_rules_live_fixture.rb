# frozen_string_literal: true

# Bounded physical FreePDK45 M2/VIA2 fixtures for the atomic live M2 rules
# transaction. Coordinates are DBU integers at 0.5 nm/DBU.

include RBA

def fixture_error(message)
  raise("M2 rules live fixture: #{message}")
end

def required_output
  value = ($output || "").to_s
  fixture_error("missing -rd output=PATH") if value.empty?
  expanded = File.expand_path(value)
  parent = File.realpath(File.dirname(expanded))
  path = File.join(parent, File.basename(expanded))
  fixture_error("refusing to overwrite output: #{path}") if
    File.exist?(path) || File.symlink?(path)
  path
end

def add_base_operands(cell, layers, metal2_box, via2_boxes)
  metal1, _via1, metal2, via2, metal3 = layers
  cell.shapes(metal2).insert(metal2_box)
  via2_boxes.each { |box| cell.shapes(via2).insert(box) }
  cell.shapes(metal3).insert(
    RBA::Box.new(
      metal2_box.left - 200,
      metal2_box.bottom - 200,
      metal2_box.right + 200,
      metal2_box.top + 200
    )
  )
  # Keep physical M1 present and comfortably legal in all-mode ownership
  # probes without introducing a VIA1 consumer.
  cell.shapes(metal1).insert(RBA::Box.new(0, 0, 1_000, 1_000))
end

def add_pair_rule_case(
  layout, name, layers, rectangle_width, rectangle_height, gap
)
  cell = layout.create_cell(name)
  left = RBA::Box.new(0, 0, rectangle_width, rectangle_height)
  right = RBA::Box.new(
    rectangle_width + gap,
    0,
    2 * rectangle_width + gap,
    rectangle_height
  )
  metal1, _via1, metal2, via2, metal3 = layers
  cell.shapes(metal1).insert(
    RBA::Box.new(-200, -200, 2 * rectangle_width + gap + 200, rectangle_height + 200)
  )
  cell.shapes(metal2).insert(left)
  cell.shapes(metal2).insert(right)
  cell.shapes(via2).insert(RBA::Box.new(70, 70, 200, 200))
  cell.shapes(metal3).insert(
    RBA::Box.new(-200, -200, 2 * rectangle_width + gap + 200, rectangle_height + 200)
  )
  cell
end

def assert_exact_boxes(cell, layer, expected, label)
  actual = []
  cell.shapes(layer).each do |shape|
    fixture_error("#{label}: expected boxes only") unless shape.is_box?
    actual << shape.box
  end
  fixture_error("#{label}: expected #{expected.length} boxes, got #{actual.length}") unless
    actual.length == expected.length
  expected.each_with_index do |box, index|
    fixture_error("#{label}: box #{index} is #{actual[index]}, expected #{box}") unless
      actual[index] == box
  end
end

output = required_output

layout = RBA::Layout.new
layout.dbu = 0.0005
layers = [
  layout.layer(11, 0), # METAL1
  layout.layer(12, 0), # VIA1
  layout.layer(13, 0), # METAL2
  layout.layer(14, 0), # VIA2
  layout.layer(15, 0)  # METAL3
]
metal1, via1, metal2, via2, metal3 = layers

clean = layout.create_cell("M2_RULES_CLEAN")
add_base_operands(
  clean,
  layers,
  RBA::Box.new(0, 0, 1_000, 1_000),
  [RBA::Box.new(200, 200, 330, 330)]
)

# M2.1: 139 DBU = 69.5 nm. VIA2 still has a clean opposite X enclosure pair.
m2_1 = layout.create_cell("M2_RULES_M2_1_HIT")
add_base_operands(
  m2_1,
  layers,
  RBA::Box.new(0, 0, 1_000, 139),
  [RBA::Box.new(300, 4, 430, 134)]
)

# M2.2: 139 DBU gap. Facing edges are shorter than M2.5's 600 DBU filter.
add_pair_rule_case(layout, "M2_RULES_M2_2_HIT", layers, 400, 400, 139)

# M2.4: VIA2 is contained, but neither axis has two 70 DBU enclosure margins.
m2_4 = layout.create_cell("M2_RULES_M2_4_HIT")
add_base_operands(
  m2_4,
  layers,
  RBA::Box.new(0, 0, 1_000, 1_000),
  [RBA::Box.new(20, 20, 150, 150)]
)

# Each cumulative width-class case is safely below only its target spacing,
# above every preceding spacing threshold, below the next width class, and
# long enough for exactly the target edge-length filter.
add_pair_rule_case(layout, "M2_RULES_M2_5_HIT", layers, 400, 700, 179)
add_pair_rule_case(layout, "M2_RULES_M2_6_HIT", layers, 700, 2_000, 539)
add_pair_rule_case(layout, "M2_RULES_M2_7_HIT", layers, 1_200, 3_800, 999)
add_pair_rule_case(layout, "M2_RULES_M2_8_HIT", layers, 2_000, 5_600, 1_799)
add_pair_rule_case(layout, "M2_RULES_M2_9_HIT", layers, 3_200, 8_200, 2_999)

# M2.3 remains independently owned. This is the clean case's exact M2/VIA2
# geometry with one added VIA1. Its left and bottom margins are each only
# 5 nm, so neither opposite pair qualifies, while the candidate M2.1/.2/.4-.9
# transaction remains all-clean.
m2_3 = layout.create_cell("M2_RULES_M2_3_OWNER")
m2_3.shapes(metal1).insert(RBA::Box.new(-200, -200, 1_200, 1_200))
m2_3.shapes(metal2).insert(RBA::Box.new(0, 0, 1_000, 1_000))
m2_3.shapes(via1).insert(RBA::Box.new(10, 10, 140, 140))
m2_3.shapes(via2).insert(RBA::Box.new(200, 200, 330, 330))
m2_3.shapes(metal3).insert(RBA::Box.new(0, 0, 1_000, 1_000))

# VIA2.1: one dimension differs by one DBU. The surrounding M2/M3 enclosure
# is otherwise clean, so the M2 transaction must not consume this owner.
via2_1 = layout.create_cell("M2_RULES_VIA2_1_OWNER")
add_base_operands(
  via2_1,
  layers,
  RBA::Box.new(0, 0, 1_000, 1_000),
  [RBA::Box.new(200, 200, 329, 330)]
)

# VIA2.2: two exact cuts have a 169 DBU = 84.5 nm axial gap.
via2_2 = layout.create_cell("M2_RULES_VIA2_2_OWNER")
add_base_operands(
  via2_2,
  layers,
  RBA::Box.new(0, 0, 1_000, 1_000),
  [
    RBA::Box.new(200, 300, 330, 430),
    RBA::Box.new(499, 300, 629, 430)
  ]
)

# VIA2.3: one DBU lies outside M2 on the left, while the opposite top/bottom
# projection margins remain ample so M2.4 itself stays clean.
via2_3 = layout.create_cell("M2_RULES_VIA2_3_OWNER")
via2_3.shapes(metal1).insert(RBA::Box.new(0, 0, 1_200, 1_000))
via2_3.shapes(metal2).insert(RBA::Box.new(101, 0, 1_100, 1_000))
via2_3.shapes(via2).insert(RBA::Box.new(100, 300, 230, 430))
via2_3.shapes(metal3).insert(RBA::Box.new(0, 0, 1_200, 1_000))

# VIA2.4: M2 and the complete M2 rule transaction remain clean, but M3 is
# deliberately disjoint from the otherwise qualified VIA2.
via2_4 = layout.create_cell("M2_RULES_VIA2_4_OWNER")
via2_4.shapes(metal1).insert(RBA::Box.new(0, 0, 1_000, 1_000))
via2_4.shapes(metal2).insert(RBA::Box.new(0, 0, 1_000, 1_000))
via2_4.shapes(via2).insert(RBA::Box.new(300, 300, 430, 430))
via2_4.shapes(metal3).insert(RBA::Box.new(2_000, 2_000, 3_000, 3_000))

expected_tops = %w[
  M2_RULES_CLEAN
  M2_RULES_M2_1_HIT
  M2_RULES_M2_2_HIT
  M2_RULES_M2_3_OWNER
  M2_RULES_M2_4_HIT
  M2_RULES_M2_5_HIT
  M2_RULES_M2_6_HIT
  M2_RULES_M2_7_HIT
  M2_RULES_M2_8_HIT
  M2_RULES_M2_9_HIT
  M2_RULES_VIA2_1_OWNER
  M2_RULES_VIA2_2_OWNER
  M2_RULES_VIA2_3_OWNER
  M2_RULES_VIA2_4_OWNER
].sort
actual_tops = layout.top_cells.map(&:name).sort
fixture_error("unexpected top cells: #{actual_tops.join(',')}") unless
  actual_tops == expected_tops

deck_clean_m2 = RBA::Box.new(0, 0, 1_000, 1_000)
deck_clean_via2 = RBA::Box.new(200, 200, 330, 330)
assert_exact_boxes(clean, metal2, [deck_clean_m2], "deck-clean M2")
assert_exact_boxes(clean, via2, [deck_clean_via2], "deck-clean VIA2")
assert_exact_boxes(m2_3, metal2, [deck_clean_m2], "M2.3-owner M2")
assert_exact_boxes(m2_3, via2, [deck_clean_via2], "M2.3-owner VIA2")
assert_exact_boxes(
  m2_3,
  via1,
  [RBA::Box.new(10, 10, 140, 140)],
  "M2.3-owner VIA1"
)
assert_exact_boxes(via2_1, metal2, [deck_clean_m2], "VIA2.1-owner M2")
assert_exact_boxes(
  via2_1,
  via2,
  [RBA::Box.new(200, 200, 329, 330)],
  "VIA2.1-owner VIA2"
)

layout.write(output)
puts(
  "M2_RULES_LIVE_FIXTURE ok path=#{output} tops=#{expected_tops.length} " \
  "dbu=#{layout.dbu} metal2=13/0 via2=14/0 " \
  "deck_clean=M2[0,0;1000,1000]+VIA2[200,200;330,330] " \
  "m2_3_owner=VIA1[10,10;140,140] via2_owner=VIA2.1"
)
