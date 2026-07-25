# frozen_string_literal: true

# Exact KLayout Boolean-equivalence gate for the small-first FreePDK45 antenna
# diode expression.  Keep the unparenthesized historical expression below:
# the first XOR also proves Ruby's actual operator grouping.

include RBA

input_path = File.expand_path(($input || "").to_s)
raise("missing -rd input=PATH") if input_path.empty?
raise("input layout is missing: #{input_path}") unless File.file?(input_path)

layout = RBA::Layout.new
layout.read(input_path)
top = layout.cell("ANTENNA_DIODE_REASSOCIATION")
raise("missing ANTENNA_DIODE_REASSOCIATION top") if top.nil?
raise("fixture has multiple top cells") unless layout.top_cells == [top]

active_layer = layout.find_layer(1, 0)
nwell_layer = layout.find_layer(3, 0)
nplus_layer = layout.find_layer(4, 0)
raise("fixture layer is missing") if [active_layer, nwell_layer, nplus_layer].any? { |i| i < 0 }

def layer_region(cell, layer)
  RBA::Region.new(cell.begin_shapes_rec(layer))
end

def diode_forms(cell, active_layer, nwell_layer, nplus_layer)
  active = layer_region(cell, active_layer)
  nwell = layer_region(cell, nwell_layer)
  nplus = layer_region(cell, nplus_layer)

  # This is the exact unparenthesized expression in the historical deck.
  historical_implicit = nplus & active - nwell
  historical_explicit = nplus & (active - nwell)
  reassociated = (nplus & active) - nwell

  [active, nwell, nplus, historical_implicit, historical_explicit, reassociated]
end

def assert_empty_xor(label, lhs, rhs)
  difference = lhs.xor(rhs)
  return if difference.is_empty?

  raise(
    "#{label}: symmetric difference is nonempty" \
    " polygons=#{difference.size} area=#{difference.area}"
  )
end

active, nwell, nplus, implicit, explicit, reassociated =
  diode_forms(top, active_layer, nwell_layer, nplus_layer)

raise("historical diode result is unexpectedly empty") if implicit.is_empty?
raise("partial-WELL overlap is missing") if (nplus & active & nwell).is_empty?
assert_empty_xor("historical parser grouping", implicit, explicit)
assert_empty_xor("small-first reassociation", explicit, reassociated)

{
  "ANTENNA_DIODE_EMPTY_NPLUS" => true,
  "ANTENNA_DIODE_EMPTY_ACTIVE" => true,
  "ANTENNA_DIODE_DISJOINT" => true,
  "ANTENNA_DIODE_EMPTY_NWELL" => false,
  "ANTENNA_DIODE_PARENT_CHILD_OVERLAP" => false
}.each do |name, expected_empty|
  cell = layout.cell(name)
  raise("missing fixture cell #{name}") if cell.nil?
  _, _, _, local_implicit, local_explicit, local_reassociated =
    diode_forms(cell, active_layer, nwell_layer, nplus_layer)
  assert_empty_xor("#{name} parser grouping", local_implicit, local_explicit)
  assert_empty_xor("#{name} reassociation", local_explicit, local_reassociated)
  if local_implicit.is_empty? != expected_empty
    raise("#{name}: unexpected empty result")
  end
end

# Stable census: besides making the proof inspectable, these values prevent a
# vacuous fixture edit from silently weakening the gate.
expected = {
  active_polygons: 23,
  active_area: 8_850_000,
  nwell_polygons: 17,
  nwell_area: 2_610_000,
  nplus_polygons: 17,
  nplus_area: 4_410_000,
  diode_polygons: 29,
  diode_area: 2_740_000
}
actual = {
  active_polygons: active.size,
  active_area: active.area,
  nwell_polygons: nwell.size,
  nwell_area: nwell.area,
  nplus_polygons: nplus.size,
  nplus_area: nplus.area,
  diode_polygons: implicit.size,
  diode_area: implicit.area
}
raise("fixture census mismatch: expected=#{expected} actual=#{actual}") unless actual == expected

puts(
  "ANTENNA_DIODE_REASSOCIATION_GATE PASS" \
  " active_polygons=#{actual[:active_polygons]}" \
  " active_area=#{actual[:active_area]}" \
  " nwell_polygons=#{actual[:nwell_polygons]}" \
  " nwell_area=#{actual[:nwell_area]}" \
  " nplus_polygons=#{actual[:nplus_polygons]}" \
  " nplus_area=#{actual[:nplus_area]}" \
  " diode_polygons=#{actual[:diode_polygons]}" \
  " diode_area=#{actual[:diode_area]}" \
  " parser_xor=empty reassociation_xor=empty"
)
