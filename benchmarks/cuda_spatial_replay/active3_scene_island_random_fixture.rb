# frozen_string_literal: true

# Deterministic small scene for CPU-brute-force versus CUDA spatial-index
# differential testing.  Geometry straddles many 2000-DBU grid boundaries and
# is instantiated through every simple orthogonal transform.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005
well_layer = layout.layer(101, 0)
active_layer = layout.layer(102, 0)
child = layout.create_cell("ACTIVE3_RANDOM_CHILD")
top = layout.create_cell("KLAYOUT_CUDA_ACTIVE3_SCENE")
random = Random.new(0xA3C0DA)
gaps = [25, 50, 109, 110, 111, 175].freeze

24.times do |index|
  column = index % 6
  row = index / 6
  x = column * 5200 + random.rand(250..1750)
  y = row * 5200 + random.rand(250..1750)
  width = random.rand(1000..1800)
  height = random.rand(1000..1800)
  well = RBA::Box.new(x, y, x + width, y + height)
  gap = gaps.fetch(index % gaps.length)
  raise("fixture gap is too large") if gap * 2 >= [width, height].min
  active = RBA::Box.new(
    x + gap, y + gap, x + width - gap, y + height - gap
  )
  child.shapes(well_layer).insert(well)
  child.shapes(active_layer).insert(active)
end

transforms = [
  RBA::Trans::R0, RBA::Trans::R90, RBA::Trans::R180, RBA::Trans::R270,
  RBA::Trans::M0, RBA::Trans::M45, RBA::Trans::M90, RBA::Trans::M135
]
transforms.each_with_index do |transform, index|
  tx = (index % 4) * 100_000 + 50_000
  ty = (index / 4) * 100_000 + 50_000
  top.insert(
    RBA::CellInstArray.new(
      child.cell_index, RBA::Trans.new(transform, tx, ty)
    )
  )
end

layout.write(output)
puts(
  "ACTIVE3_SCENE_ISLAND_RANDOM_FIXTURE ok " \
  "path=#{output} seed=0xA3C0DA boxes=24 transforms=8"
)
