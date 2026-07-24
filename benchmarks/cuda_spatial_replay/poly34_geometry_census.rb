# frozen_string_literal: true

# Read-only production census for the exact derived `gate = poly & active`
# operand used by POLY.3/POLY.4.  The first certificate milestone deliberately
# supports rectangle gates only, so any non-box, hole, or non-Manhattan result
# is a correctness NO_GO before host/ABI/live integration.

include RBA

def census_error(message)
  raise("POLY34 geometry census: #{message}")
end

def required_rd(name, value)
  census_error("missing -rd #{name}=VALUE") if value.nil? || value.to_s.empty?
  value.to_s
end

input_path = File.realpath(required_rd("input", $input))
top_name = required_rd("topcell", $topcell)
poly_layer_number = required_rd("poly_layer", $poly_layer).to_i
poly_datatype = required_rd("poly_datatype", $poly_datatype).to_i
active_layer_number = required_rd("active_layer", $active_layer).to_i
active_datatype = required_rd("active_datatype", $active_datatype).to_i

layout = RBA::Layout.new
layout.read(input_path)
top = layout.cell(top_name)
census_error("missing top cell #{top_name}") if top.nil?

poly_layer = layout.layer(poly_layer_number, poly_datatype)
active_layer = layout.layer(active_layer_number, active_datatype)

reachable = {}
stack = [top]
until stack.empty?
  cell = stack.pop
  next if reachable[cell.cell_index]
  reachable[cell.cell_index] = cell
  cell.each_inst { |instance| stack << layout.cell(instance.cell_index) }
end

def template_census(cells, layer)
  counts = Hash.new(0)
  cells.each_value do |cell|
    cell.shapes(layer).each do |shape|
      counts[:records] += 1
      if shape.is_box?
        counts[:boxes] += 1
      elsif shape.is_polygon? || shape.is_simple_polygon?
        polygon = shape.is_polygon? ? shape.polygon : shape.simple_polygon
        counts[polygon.is_rectilinear? ? :manhattan_polygons : :non_manhattan_polygons] += 1
        counts[:polygons_with_holes] += 1 unless polygon.holes.zero?
      elsif shape.is_path?
        counts[:paths] += 1
      elsif shape.is_text?
        counts[:texts] += 1
      else
        counts[:other_geometry] += 1
      end
    end
  end
  counts
end

poly_templates = template_census(reachable, poly_layer)
active_templates = template_census(reachable, active_layer)
poly = RBA::Region.new(top.begin_shapes_rec(poly_layer))
active = RBA::Region.new(top.begin_shapes_rec(active_layer))
gate = poly & active

counts = Hash.new(0)
widths = Hash.new(0)
heights = Hash.new(0)
gate.each_merged do |polygon|
  counts[:total] += 1
  counts[:with_holes] += 1 unless polygon.holes.zero?
  if polygon.is_box?
    counts[:boxes] += 1
    box = polygon.bbox
    widths[box.width] += 1
    heights[box.height] += 1
  elsif polygon.is_rectilinear?
    counts[:manhattan_nonboxes] += 1
  else
    counts[:non_manhattan] += 1
  end
end

qualified =
  counts[:total].positive? &&
  counts[:boxes] == counts[:total] &&
  counts[:manhattan_nonboxes].zero? &&
  counts[:non_manhattan].zero? &&
  counts[:with_holes].zero? &&
  [poly_templates, active_templates].all? do |templates|
    templates[:non_manhattan_polygons].zero? &&
      templates[:polygons_with_holes].zero? &&
      templates[:paths].zero? &&
      templates[:other_geometry].zero?
  end
puts(
  "POLY34_GEOMETRY_CENSUS verdict=#{qualified ? 'GO' : 'NO_GO'} " \
  "top=#{top_name} dbu=#{layout.dbu} " \
  "poly_flat_records=#{poly.count} active_flat_records=#{active.count} " \
  "gate_merged=#{counts[:total]} gate_boxes=#{counts[:boxes]} " \
  "gate_manhattan_nonboxes=#{counts[:manhattan_nonboxes]} " \
  "gate_non_manhattan=#{counts[:non_manhattan]} " \
  "gate_with_holes=#{counts[:with_holes]}"
)
[
  ["poly", poly_templates],
  ["active", active_templates]
].each do |name, templates|
  puts(
    "POLY34_PRIMARY_TEMPLATES layer=#{name} " \
    "records=#{templates[:records]} boxes=#{templates[:boxes]} " \
    "manhattan_polygons=#{templates[:manhattan_polygons]} " \
    "non_manhattan_polygons=#{templates[:non_manhattan_polygons]} " \
    "polygons_with_holes=#{templates[:polygons_with_holes]} " \
    "paths=#{templates[:paths]} texts=#{templates[:texts]} " \
    "other_geometry=#{templates[:other_geometry]}"
  )
end
puts(
  "POLY34_GATE_WIDTHS " +
  widths.sort_by { |width, count| [-count, width] }.first(20).map { |width, count|
    "#{width}:#{count}"
  }.join(",")
)
puts(
  "POLY34_GATE_HEIGHTS " +
  heights.sort_by { |height, count| [-count, height] }.first(20).map { |height, count|
    "#{height}:#{count}"
  }.join(",")
)
exit(qualified ? 0 : 1)
