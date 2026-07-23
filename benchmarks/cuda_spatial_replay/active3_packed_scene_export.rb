# frozen_string_literal: true

# Export a captured ACTIVE.3 derived GDS into the pointer-free KACTSCN1 format.
#
# Run with:
#   klayout -b -r active3_packed_scene_export.rb \
#     -rd input=scene.gds -rd topcell=KLAYOUT_CUDA_ACTIVE3_SCENE \
#     -rd output=scene.kact
#
# The selected input layers default to 101/0 (WELL) and 102/0 (ACTIVE).

require "digest"
require "tempfile"
include RBA

MAGIC = "KACTSCN\0".b
FORMAT_VERSION = 1
HEADER_BYTES = 256
ENDIAN_TAG = 0x0102_0304
FORMAT_FLAGS = 1 # Bounding boxes contain inclusive coordinate extrema.
COORDINATE_BITS = 64
LAYER_COUNT = 2
CELL_RECORD_BYTES = 208
INSTANCE_RECORD_BYTES = 96
POLYGON_RECORD_BYTES = 64
EDGE_RECORD_BYTES = 56

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

def scene_error(message)
  raise("ACTIVE.3 packed-scene export: #{message}")
end

def required_rd(name, value)
  scene_error("missing -rd #{name}=VALUE") if value.nil? || value.to_s.empty?
  value.to_s
end

def optional_nonnegative_rd(name, value, default)
  text = value.nil? || value.to_s.empty? ? default.to_s : value.to_s
  number = Integer(text, 10)
  scene_error("#{name} must be nonnegative") if number.negative?
  scene_error("#{name} exceeds uint32") if number > UINT32_MAX
  number
rescue ArgumentError
  scene_error("#{name} must be a base-10 integer")
end

def checked_i64(value, what)
  number = Integer(value)
  scene_error("#{what} is outside signed int64") if number < INT64_MIN || number > INT64_MAX
  number
end

def checked_u32(value, what)
  number = Integer(value)
  scene_error("#{what} is outside uint32") if number.negative? || number > UINT32_MAX
  number
end

def checked_u64(value, what)
  number = Integer(value)
  scene_error("#{what} is outside uint64") if number.negative? || number > UINT64_MAX
  number
end

def checked_add_i64(a, b, what)
  checked_i64(a + b, what)
end

def checked_mul_i64(a, b, what)
  checked_i64(a * b, what)
end

def align_up(value, alignment)
  scene_error("invalid alignment") unless alignment.positive? && (alignment & (alignment - 1)).zero?
  checked_u64((value + alignment - 1) & -alignment, "aligned file offset")
end

def bbox_from_vertices(vertices)
  xs = vertices.map(&:first)
  ys = vertices.map(&:last)
  [xs.min, ys.min, xs.max, ys.max]
end

def bbox_union(a, b)
  return b.dup if a.nil?
  return a.dup if b.nil?
  [
    [a[0], b[0]].min,
    [a[1], b[1]].min,
    [a[2], b[2]].max,
    [a[3], b[3]].max
  ]
end

def manhattan_segments_intersect?(first, second)
  ax1, ay1, ax2, ay2 = first
  bx1, by1, bx2, by2 = second
  if ay1 == ay2 && by1 == by2
    alo, ahi = [ax1, ax2].minmax
    blo, bhi = [bx1, bx2].minmax
    ay1 == by1 && [alo, blo].max <= [ahi, bhi].min
  elsif ax1 == ax2 && bx1 == bx2
    alo, ahi = [ay1, ay2].minmax
    blo, bhi = [by1, by2].minmax
    ax1 == bx1 && [alo, blo].max <= [ahi, bhi].min
  else
    horizontal, vertical = ay1 == ay2 ? [first, second] : [second, first]
    hx1, hy, hx2, = horizontal
    vx, vy1, _, vy2 = vertical
    hxlo, hxhi = [hx1, hx2].minmax
    vylo, vyhi = [vy1, vy2].minmax
    hxlo <= vx && vx <= hxhi && vylo <= hy && hy <= vyhi
  end
end

def positive_collinear_overlap?(first, second)
  ax1, ay1, ax2, ay2 = first
  bx1, by1, bx2, by2 = second
  if ay1 == ay2 && by1 == by2 && ay1 == by1
    alo, ahi = [ax1, ax2].minmax
    blo, bhi = [bx1, bx2].minmax
    [ahi, bhi].min > [alo, blo].max
  elsif ax1 == ax2 && bx1 == bx2 && ax1 == bx1
    alo, ahi = [ay1, ay2].minmax
    blo, bhi = [by1, by2].minmax
    [ahi, bhi].min > [alo, blo].max
  else
    false
  end
end

def transform_point(code, x, y, what = "transformed point")
  tx, ty =
    case code
    when 0 then [x, y]
    when 1 then [-y, x]
    when 2 then [-x, -y]
    when 3 then [y, -x]
    when 4 then [x, -y]
    when 5 then [y, x]
    when 6 then [-x, y]
    when 7 then [-y, -x]
    else
      scene_error("invalid orthogonal transform code #{code}")
    end
  # Ruby integers do not overflow. Check the raw orthogonal transform before
  # translation so an int64 consumer cannot encounter -INT64_MIN even when a
  # following displacement would bring the final coordinate back in range.
  [
    checked_i64(tx, "#{what} x before translation"),
    checked_i64(ty, "#{what} y before translation")
  ]
end

def transformed_array_bbox(box, instance)
  corners = [
    [box[0], box[1]], [box[0], box[3]],
    [box[2], box[1]], [box[2], box[3]]
  ].map do |x, y|
    tx, ty = transform_point(
      instance[:transform], x, y, "child subtree bbox corner"
    )
    [
      checked_add_i64(tx, instance[:dx], "transformed x coordinate"),
      checked_add_i64(ty, instance[:dy], "transformed y coordinate")
    ]
  end
  base = bbox_from_vertices(corners)

  a_last_x = checked_mul_i64(instance[:columns] - 1, instance[:ax], "array column x extent")
  a_last_y = checked_mul_i64(instance[:columns] - 1, instance[:ay], "array column y extent")
  b_last_x = checked_mul_i64(instance[:rows] - 1, instance[:bx], "array row x extent")
  b_last_y = checked_mul_i64(instance[:rows] - 1, instance[:by], "array row y extent")
  offsets_x = [
    0, a_last_x, b_last_x,
    checked_add_i64(a_last_x, b_last_x, "combined array x extent")
  ]
  offsets_y = [
    0, a_last_y, b_last_y,
    checked_add_i64(a_last_y, b_last_y, "combined array y extent")
  ]

  [
    checked_add_i64(base[0], offsets_x.min, "array bbox left"),
    checked_add_i64(base[1], offsets_y.min, "array bbox bottom"),
    checked_add_i64(base[2], offsets_x.max, "array bbox right"),
    checked_add_i64(base[3], offsets_y.max, "array bbox top")
  ]
end

def canonical_contour(shape, cell_name, layer_name)
  scene_error("#{cell_name} #{layer_name}: shape has properties") unless shape.prop_id == 0
  unless shape.is_box? || shape.is_polygon? || shape.is_simple_polygon?
    kind =
      if shape.is_path?
        "path"
      elsif shape.is_text?
        "text"
      elsif shape.is_edge?
        "edge"
      else
        "non-polygon"
      end
    scene_error("#{cell_name} #{layer_name}: unsupported #{kind} shape")
  end

  polygon = shape.polygon
  scene_error("#{cell_name} #{layer_name}: polygon has holes") unless polygon.holes == 0

  edges = []
  polygon.each_edge do |edge|
    x1 = checked_i64(edge.p1.x, "#{cell_name} #{layer_name} edge x1")
    y1 = checked_i64(edge.p1.y, "#{cell_name} #{layer_name} edge y1")
    x2 = checked_i64(edge.p2.x, "#{cell_name} #{layer_name} edge x2")
    y2 = checked_i64(edge.p2.y, "#{cell_name} #{layer_name} edge y2")
    scene_error("#{cell_name} #{layer_name}: degenerate edge") if x1 == x2 && y1 == y2
    scene_error("#{cell_name} #{layer_name}: non-Manhattan edge") unless x1 == x2 || y1 == y2
    edges << [x1, y1, x2, y2]
  end
  scene_error("#{cell_name} #{layer_name}: contour has fewer than four edges") if edges.length < 4

  edges.each_with_index do |edge, index|
    following = edges[(index + 1) % edges.length]
    unless edge[2] == following[0] && edge[3] == following[1]
      scene_error("#{cell_name} #{layer_name}: malformed open contour")
    end
  end

  vertices = edges.map { |edge| [edge[0], edge[1]] }
  scene_error("#{cell_name} #{layer_name}: contour repeats a vertex") unless vertices.uniq.length == vertices.length

  # Unique vertices do not prove a simple polygon: non-adjacent Manhattan
  # segments can cross, and adjacent collinear segments can backtrack.
  edges.each_index do |first|
    ((first + 1)...edges.length).each do |second|
      next unless manhattan_segments_intersect?(edges[first], edges[second])
      adjacent = second == first + 1 ||
                 (first.zero? && second == edges.length - 1)
      next if adjacent &&
              !positive_collinear_overlap?(edges[first], edges[second])
      scene_error(
        "#{cell_name} #{layer_name}: self-intersecting contour at edges " \
        "#{first} and #{second}"
      )
    end
  end

  twice_area = 0
  vertices.each_with_index do |point, index|
    following = vertices[(index + 1) % vertices.length]
    twice_area += point[0] * following[1] - following[0] * point[1]
  end
  scene_error("#{cell_name} #{layer_name}: outer contour is not clockwise") unless twice_area.negative?

  # Rotate, but never reverse, the directed contour. This makes serialization
  # independent of the iterator's cyclic start while retaining edge direction.
  rotations = vertices.length.times.map { |index| vertices.rotate(index) }
  canonical_vertices = rotations.min
  canonical_edges = canonical_vertices.each_with_index.map do |point, index|
    following = canonical_vertices[(index + 1) % canonical_vertices.length]
    [point[0], point[1], following[0], following[1]]
  end
  {
    vertices: canonical_vertices,
    edges: canonical_edges,
    bbox: bbox_from_vertices(canonical_vertices)
  }
end

input_expanded = File.expand_path(required_rd("input", $input))
top_name = required_rd("topcell", $topcell)
output_expanded = File.expand_path(required_rd("output", $output))
well_layer_number = optional_nonnegative_rd("well_layer", $well_layer, 101)
well_datatype = optional_nonnegative_rd("well_datatype", $well_datatype, 0)
active_layer_number = optional_nonnegative_rd("active_layer", $active_layer, 102)
active_datatype = optional_nonnegative_rd("active_datatype", $active_datatype, 0)
if well_layer_number == active_layer_number && well_datatype == active_datatype
  scene_error("WELL and ACTIVE input layers must be distinct")
end

scene_error("input does not exist: #{input_expanded}") unless File.file?(input_expanded)
input_path = File.realpath(input_expanded)
output_parent = File.realpath(File.dirname(output_expanded))
output_path = File.join(output_parent, File.basename(output_expanded))
scene_error("output aliases input") if output_path == input_path
scene_error("refusing to overwrite output: #{output_path}") if File.exist?(output_path) || File.symlink?(output_path)

layout = RBA::Layout.new
layout.read(input_path)
scene_error("layout database unit must be finite and positive") unless layout.dbu.finite? && layout.dbu.positive?
top = layout.cell(top_name)
scene_error("missing top cell #{top_name}") unless top

layers = [
  {
    name: "WELL",
    number: well_layer_number,
    datatype: well_datatype,
    index: layout.find_layer(RBA::LayerInfo.new(well_layer_number, well_datatype))
  },
  {
    name: "ACTIVE",
    number: active_layer_number,
    datatype: active_datatype,
    index: layout.find_layer(RBA::LayerInfo.new(active_layer_number, active_datatype))
  }
]
layers.each { |layer| scene_error("missing #{layer[:name]} layer") if layer[:index].nil? }

reachable = {}
stack = [top]
until stack.empty?
  cell = stack.pop
  next if reachable.key?(cell.cell_index)
  reachable[cell.cell_index] = cell
  cell.each_inst do |instance|
    child = layout.cell(instance.cell_index)
    scene_error("#{cell.name}: instance refers to missing cell #{instance.cell_index}") unless child
    stack << child
  end
end
scene_error("reachable hierarchy is empty") if reachable.empty?

cells = reachable.values.sort_by { |cell| [cell.name.to_s.b, cell.cell_index] }
cell_names = cells.map { |cell| cell.name.to_s.b }
scene_error("reachable cells have duplicate names") unless cell_names.uniq.length == cell_names.length
source_to_dense = {}
cells.each_with_index do |cell, dense_id|
  checked_u64(dense_id, "cell ID")
  source_to_dense[cell.cell_index] = dense_id
end
root_cell_id = source_to_dense.fetch(top.cell_index)

names_data = +"".b
cell_data = cells.each_with_index.map do |cell, cell_id|
  name = cell.name.to_s.b
  scene_error("cell #{cell_id} has an empty name") if name.empty?
  scene_error("cell name contains NUL") if name.include?("\0")
  checked_u32(name.bytesize, "#{cell.name} name length")
  name_offset = names_data.bytesize
  names_data << name
  {
    id: cell_id,
    source: cell,
    name: name,
    name_offset: name_offset,
    instances: [],
    polygons: [],
    local_bboxes: [nil, nil],
    subtree_bboxes: nil
  }
end

instances = []
polygons = []
edges = []

cell_data.each do |record|
  cell = record[:source]
  local_instances = []
  cell.each_inst do |instance|
    scene_error("#{cell.name}: instance has properties") unless instance.prop_id == 0
    scene_error("#{cell.name}: complex instance transform") if instance.is_complex?
    child_id = source_to_dense[instance.cell_index]
    scene_error("#{cell.name}: instance child is unreachable") if child_id.nil?

    size = checked_u64(instance.size, "#{cell.name} instance size")
    if instance.is_regular_array?
      columns = [checked_u32(instance.na, "#{cell.name} array na"), 1].max
      rows = [checked_u32(instance.nb, "#{cell.name} array nb"), 1].max
      scene_error("#{cell.name}: malformed regular-array size") unless size == columns * rows
      ax = checked_i64(instance.a.x, "#{cell.name} array ax")
      ay = checked_i64(instance.a.y, "#{cell.name} array ay")
      bx = checked_i64(instance.b.x, "#{cell.name} array bx")
      by = checked_i64(instance.b.y, "#{cell.name} array by")
    else
      scene_error("#{cell.name}: unsupported irregular instance array") unless size == 1
      columns = 1
      rows = 1
      ax = ay = bx = by = 0
    end

    if columns == 1
      ax = ay = 0
    elsif ax.zero? && ay.zero?
      scene_error("#{cell.name}: repeated array columns have zero pitch")
    end
    if rows == 1
      bx = by = 0
    elsif bx.zero? && by.zero?
      scene_error("#{cell.name}: repeated array rows have zero pitch")
    end

    trans = instance.trans
    transform = checked_u32(trans.rot, "#{cell.name} transform code")
    scene_error("#{cell.name}: invalid orthogonal transform code") unless transform < 8
    dx = checked_i64(trans.disp.x, "#{cell.name} instance dx")
    dy = checked_i64(trans.disp.y, "#{cell.name} instance dy")
    a_last_x = checked_mul_i64(columns - 1, ax, "#{cell.name} array column x extent")
    a_last_y = checked_mul_i64(columns - 1, ay, "#{cell.name} array column y extent")
    b_last_x = checked_mul_i64(rows - 1, bx, "#{cell.name} array row x extent")
    b_last_y = checked_mul_i64(rows - 1, by, "#{cell.name} array row y extent")
    [
      [0, 0], [a_last_x, a_last_y], [b_last_x, b_last_y],
      [
        checked_add_i64(a_last_x, b_last_x, "#{cell.name} combined array x extent"),
        checked_add_i64(a_last_y, b_last_y, "#{cell.name} combined array y extent")
      ]
    ].each do |offset_x, offset_y|
      checked_add_i64(dx, offset_x, "#{cell.name} array origin x")
      checked_add_i64(dy, offset_y, "#{cell.name} array origin y")
    end

    local_instances << {
      parent: record[:id],
      child: child_id,
      occurrences: checked_u64(columns * rows, "#{cell.name} array occurrences"),
      dx: dx, dy: dy, ax: ax, ay: ay, bx: bx, by: by,
      columns: columns, rows: rows, transform: transform
    }
  end
  local_instances.sort_by! do |instance|
    [
      instance[:child], instance[:transform], instance[:dx], instance[:dy],
      instance[:columns], instance[:rows],
      instance[:ax], instance[:ay], instance[:bx], instance[:by]
    ]
  end
  local_instances.each do |instance|
    instance[:id] = checked_u64(instances.length, "instance ID")
    instances << instance
  end
  record[:instances] = local_instances

  local_polygons = []
  layers.each_with_index do |layer, layer_code|
    cell.shapes(layer[:index]).each do |shape|
      contour = canonical_contour(shape, cell.name, layer[:name])
      local_polygons << contour.merge(layer: layer_code)
    end
  end
  local_polygons.sort_by! do |polygon|
    [polygon[:layer], polygon[:vertices].flatten]
  end
  local_polygons.each do |polygon|
    polygon[:id] = checked_u64(polygons.length, "polygon ID")
    polygon[:cell] = record[:id]
    polygon[:edge_begin] = checked_u64(edges.length, "polygon edge offset")
    polygon[:edges].each_with_index do |coordinates, local_index|
      edges << {
        id: checked_u64(edges.length, "edge ID"),
        polygon: polygon[:id],
        coordinates: coordinates,
        local_index: checked_u32(local_index, "local edge index"),
        layer: polygon[:layer]
      }
    end
    record[:local_bboxes][polygon[:layer]] =
      bbox_union(record[:local_bboxes][polygon[:layer]], polygon[:bbox])
    polygons << polygon
  end
  record[:polygons] = local_polygons
end

visit_state = Array.new(cell_data.length, 0)
compute_subtree = lambda do |cell_id|
  state = visit_state[cell_id]
  scene_error("hierarchy cycle through cell #{cell_data[cell_id][:name]}") if state == 1
  return cell_data[cell_id][:subtree_bboxes] if state == 2
  visit_state[cell_id] = 1

  boxes = cell_data[cell_id][:local_bboxes].map { |box| box&.dup }
  cell_data[cell_id][:instances].each do |instance|
    child_boxes = compute_subtree.call(instance[:child])
    child_boxes.each_with_index do |child_box, layer_code|
      next if child_box.nil?
      transformed = transformed_array_bbox(child_box, instance)
      boxes[layer_code] = bbox_union(boxes[layer_code], transformed)
    end
  end

  cell_data[cell_id][:subtree_bboxes] = boxes
  visit_state[cell_id] = 2
  boxes
end
compute_subtree.call(root_cell_id)
cell_data.each_index { |cell_id| compute_subtree.call(cell_id) }

cell_bytes = +"".b
instance_begin = 0
polygon_begin = 0
edge_begin = 0
cell_data.each do |record|
  local_mask = record[:local_bboxes].each_index.reduce(0) do |mask, layer_code|
    mask | (record[:local_bboxes][layer_code].nil? ? 0 : (1 << layer_code))
  end
  subtree_mask = record[:subtree_bboxes].each_index.reduce(0) do |mask, layer_code|
    mask | (record[:subtree_bboxes][layer_code].nil? ? 0 : (1 << layer_code))
  end
  bbox_values = (record[:local_bboxes] + record[:subtree_bboxes]).flat_map do |box|
    box || [0, 0, 0, 0]
  end
  record_bytes = [
    record[:id], record[:name_offset],
    record[:name].bytesize, local_mask, subtree_mask, 0,
    instance_begin, record[:instances].length,
    polygon_begin, record[:polygons].length,
    edge_begin, record[:polygons].sum { |polygon| polygon[:edges].length }
  ].pack("Q<Q<L<L<L<L<Q<Q<Q<Q<Q<Q<")
  record_bytes << bbox_values.pack("q<16")
  scene_error("internal cell record size mismatch") unless record_bytes.bytesize == CELL_RECORD_BYTES
  cell_bytes << record_bytes
  instance_begin += record[:instances].length
  polygon_begin += record[:polygons].length
  edge_begin += record[:polygons].sum { |polygon| polygon[:edges].length }
end

instance_bytes = +"".b
instances.each do |instance|
  record_bytes = [
    instance[:id], instance[:parent], instance[:child], instance[:occurrences],
    instance[:dx], instance[:dy],
    instance[:ax], instance[:ay], instance[:bx], instance[:by],
    instance[:columns], instance[:rows], instance[:transform], 0
  ].pack("Q<Q<Q<Q<q<q<q<q<q<q<L<L<L<L<")
  scene_error("internal instance record size mismatch") unless record_bytes.bytesize == INSTANCE_RECORD_BYTES
  instance_bytes << record_bytes
end

polygon_bytes = +"".b
polygons.each do |polygon|
  record_bytes = [
    polygon[:id], polygon[:cell], polygon[:edge_begin],
    polygon[:edges].length, polygon[:layer]
  ].pack("Q<Q<Q<L<L<")
  record_bytes << polygon[:bbox].pack("q<4")
  scene_error("internal polygon record size mismatch") unless record_bytes.bytesize == POLYGON_RECORD_BYTES
  polygon_bytes << record_bytes
end

edge_bytes = +"".b
edges.each do |edge|
  record_bytes = [
    edge[:id], edge[:polygon], *edge[:coordinates],
    edge[:local_index], edge[:layer]
  ].pack("Q<Q<q<q<q<q<L<L<")
  scene_error("internal edge record size mismatch") unless record_bytes.bytesize == EDGE_RECORD_BYTES
  edge_bytes << record_bytes
end

checked_u64(cell_data.length, "cell count")
checked_u64(instances.length, "instance count")
checked_u64(polygons.length, "polygon count")
checked_u64(edges.length, "edge count")

names_offset = HEADER_BYTES
names_bytes = names_data.bytesize
cells_offset = align_up(names_offset + names_bytes, 64)
cells_bytes = cell_bytes.bytesize
instances_offset = align_up(cells_offset + cells_bytes, 64)
instances_bytes = instance_bytes.bytesize
polygons_offset = align_up(instances_offset + instances_bytes, 64)
polygons_bytes = polygon_bytes.bytesize
edges_offset = align_up(polygons_offset + polygons_bytes, 64)
edges_bytes = edge_bytes.bytesize
file_bytes = align_up(edges_offset + edges_bytes, 64)
payload_offset = HEADER_BYTES
payload_bytes = file_bytes - payload_offset

payload = +"".b
append_at = lambda do |absolute_offset, data|
  expected_relative = absolute_offset - payload_offset
  scene_error("internal section overlap") if payload.bytesize > expected_relative
  payload << ("\0".b * (expected_relative - payload.bytesize))
  payload << data
end
append_at.call(names_offset, names_data)
append_at.call(cells_offset, cell_bytes)
append_at.call(instances_offset, instance_bytes)
append_at.call(polygons_offset, polygon_bytes)
append_at.call(edges_offset, edge_bytes)
payload << ("\0".b * (payload_bytes - payload.bytesize))
scene_error("internal payload size mismatch") unless payload.bytesize == payload_bytes

header_prefix = +MAGIC
header_prefix << [
  FORMAT_VERSION, HEADER_BYTES, ENDIAN_TAG, FORMAT_FLAGS,
  COORDINATE_BITS, LAYER_COUNT,
  CELL_RECORD_BYTES, INSTANCE_RECORD_BYTES,
  POLYGON_RECORD_BYTES, EDGE_RECORD_BYTES,
  well_layer_number, well_datatype,
  active_layer_number, active_datatype
].pack("L<14")
header_prefix << [layout.dbu].pack("E")
header_prefix << [
  root_cell_id,
  cell_data.length, instances.length, polygons.length, edges.length,
  names_offset, names_bytes,
  cells_offset, cells_bytes,
  instances_offset, instances_bytes,
  polygons_offset, polygons_bytes,
  edges_offset, edges_bytes,
  file_bytes, payload_offset, payload_bytes
].pack("Q<18")
scene_digest = Digest::SHA256.new
scene_digest.update(header_prefix)
scene_digest.update("\0".b * 40) # Digest field plus reserved field.
scene_digest.update(payload)
scene_sha256 = scene_digest.digest
header = header_prefix
header << scene_sha256
header << [0].pack("Q<")
scene_error("internal header size mismatch") unless header.bytesize == HEADER_BYTES

published = false
begin
  # Write and sync a private file in the destination directory, then publish
  # it with an atomic no-clobber hard link. A concurrent reader can therefore
  # observe either no final path or the complete file, never a partial write.
  # A process killed mid-write can leave only an inconsequential hidden temp.
  prefix = ".#{File.basename(output_path)}."
  Tempfile.create([prefix, ".tmp"], output_parent) do |temporary|
    temporary.binmode
    temporary.chmod(0o644)
    temporary.write(header)
    temporary.write(payload)
    temporary.flush
    temporary.fsync
    File.link(temporary.path, output_path)
    published = true
    File.open(output_parent, File::RDONLY) { |directory| directory.fsync }
  end
rescue StandardError
  # Remove only a final path published by this process. If File.link failed
  # with EEXIST, the pre-existing/concurrently published output is untouched.
  File.unlink(output_path) if published && File.exist?(output_path)
  raise
end

puts(
  "KACTSCN1 output=#{output_path} bytes=#{file_bytes} " \
  "cells=#{cell_data.length} instances=#{instances.length} " \
  "polygons=#{polygons.length} edges=#{edges.length} " \
  "root=#{root_cell_id} scene_sha256=#{scene_sha256.unpack1('H*')}"
)
