include RBA

def required_rd(name, value)
  raise("missing -rd #{name}=VALUE") if value.nil? || value.to_s.empty?
  value.to_s
end

input_expanded = File.expand_path(required_rd("input", $input))
top_name = required_rd("topcell", $topcell)
output_expanded = File.expand_path(required_rd("output", $output))
well_layer_number = Integer(required_rd("well_layer", $well_layer), 10)
active_layer_number = Integer(required_rd("active_layer", $active_layer), 10)
raise("well_layer must be nonnegative") if well_layer_number.negative?
raise("active_layer must be nonnegative") if active_layer_number.negative?

raise("input does not exist: #{input_expanded}") unless File.file?(input_expanded)
input_path = File.realpath(input_expanded)
output_path = File.join(
  File.realpath(File.dirname(output_expanded)),
  File.basename(output_expanded)
)
raise("output aliases input") if output_path == input_path
raise("refusing to overwrite output: #{output_path}") if File.exist?(output_path) || File.symlink?(output_path)

layout = RBA::Layout.new
layout.read(input_path)
top = layout.cell(top_name)
raise("missing top cell #{top_name}") unless top

layer_ids = {
  "well" => layout.find_layer(RBA::LayerInfo.new(well_layer_number, 0)),
  "active" => layout.find_layer(RBA::LayerInfo.new(active_layer_number, 0))
}
layer_ids.each do |name, layer|
  raise("missing #{name} layer") if layer.nil?
end

reachable = {}
stack = [top]
until stack.empty?
  cell = stack.pop
  next if reachable[cell.cell_index]
  reachable[cell.cell_index] = cell
  cell.each_inst { |inst| stack << layout.cell(inst.cell_index) }
end

def local_layer_stats(cell, layer)
  stats = {
    shapes: 0, boxes: 0, polygons: 0, paths: 0, other: 0,
    edges: 0, non_manhattan_edges: 0, holes: 0, properties: 0
  }

  cell.shapes(layer).each do |shape|
    stats[:shapes] += 1
    stats[:properties] += 1 if shape.prop_id != 0

    if shape.is_box?
      stats[:boxes] += 1
      stats[:edges] += 4
    elsif shape.is_polygon? || shape.is_simple_polygon?
      stats[:polygons] += 1
      polygon = shape.polygon
      stats[:holes] += polygon.holes
      polygon.each_edge do |edge|
        stats[:edges] += 1
        stats[:non_manhattan_edges] += 1 unless
          edge.p1.x == edge.p2.x || edge.p1.y == edge.p2.y
      end
    elsif shape.is_path?
      stats[:paths] += 1
      polygon = shape.polygon
      stats[:holes] += polygon.holes
      polygon.each_edge do |edge|
        stats[:edges] += 1
        stats[:non_manhattan_edges] += 1 unless
          edge.p1.x == edge.p2.x || edge.p1.y == edge.p2.y
      end
    else
      stats[:other] += 1
    end
  end

  stats
end

local = {}
reachable.each_value do |cell|
  local[cell.cell_index] = {}
  layer_ids.each do |name, layer|
    local[cell.cell_index][name] = local_layer_stats(cell, layer)
  end
end

memo = {}
visiting = {}
subtree = lambda do |cell|
  return memo[cell.cell_index] if memo[cell.cell_index]
  raise("hierarchy cycle at #{cell.name}") if visiting[cell.cell_index]
  visiting[cell.cell_index] = true

  totals = {}
  layer_ids.each_key do |name|
    totals[name] = {
      shapes: local[cell.cell_index][name][:shapes],
      edges: local[cell.cell_index][name][:edges]
    }
  end

  cell.each_inst do |inst|
    child = layout.cell(inst.cell_index)
    child_totals = subtree.call(child)
    multiplicity = [inst.na, 1].max * [inst.nb, 1].max
    layer_ids.each_key do |name|
      totals[name][:shapes] += multiplicity * child_totals[name][:shapes]
      totals[name][:edges] += multiplicity * child_totals[name][:edges]
    end
  end

  visiting.delete(cell.cell_index)
  memo[cell.cell_index] = totals
end

flat = subtree.call(top)
aggregate = {}
layer_ids.each_key do |name|
  aggregate[name] = local.values.map { |s| s[name] }.reduce({}) do |sum, stats|
    stats.each { |key, value| sum[key] = (sum[key] || 0) + value }
    sum
  end
end

instance_records = 0
array_records = 0
array_elements = 0
complex_instances = 0
property_instances = 0
max_na = 0
max_nb = 0
reachable.each_value do |cell|
  cell.each_inst do |inst|
    instance_records += 1
    multiplicity = [inst.na, 1].max * [inst.nb, 1].max
    array_records += 1 if multiplicity > 1
    array_elements += multiplicity
    complex_instances += 1 if inst.is_complex?
    property_instances += 1 if inst.prop_id != 0
    max_na = [max_na, inst.na].max
    max_nb = [max_nb, inst.nb].max
  end
end

lines = []
lines << "format=klayout-active3-scene-census-v1"
lines << "top=#{top.name} dbu=#{layout.dbu}"
lines << "cells=#{reachable.size} instance_records=#{instance_records} " \
         "array_records=#{array_records} array_elements=#{array_elements} " \
         "complex_instances=#{complex_instances} " \
         "property_instances=#{property_instances} max_na=#{max_na} max_nb=#{max_nb}"
layer_ids.each_key do |name|
  stats = aggregate[name]
  lines << "#{name} stored_shapes=#{stats[:shapes]} stored_edges=#{stats[:edges]} " \
           "boxes=#{stats[:boxes]} polygons=#{stats[:polygons]} paths=#{stats[:paths]} " \
           "other=#{stats[:other]} holes=#{stats[:holes]} " \
           "non_manhattan_edges=#{stats[:non_manhattan_edges]} " \
           "property_shapes=#{stats[:properties]} flat_shapes=#{flat[name][:shapes]} " \
           "flat_edges=#{flat[name][:edges]}"
end

exclusive_create = File::WRONLY | File::CREAT | File::EXCL
File.open(output_path, exclusive_create, 0o644) do |output|
  lines.each { |line| output.puts(line) }
end
puts(lines.join("\n"))
