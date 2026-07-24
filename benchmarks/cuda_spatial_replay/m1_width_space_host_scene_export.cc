/*
 * Standalone exact merged-M1 capture bridge.
 *
 * Reads one explicitly asserted merged polygon layer from a layout capture,
 * rebuilds the committed db::CudaM1WidthSpaceScene, and publishes its
 * canonical little-endian byte stream in KM1WSCN1 format.
 */

#include "m1_width_space_host_scene_format.h"

#include "dbCudaActive3Digest.h"
#include "dbCudaM1WidthSpace.h"
#include "dbDeepShapeStore.h"
#include "dbGDS2Reader.h"
#include "dbLayout.h"
#include "dbLoadLayoutOptions.h"
#include "dbRecursiveShapeIterator.h"
#include "tlStream.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace
{

namespace format = klayout_m1ws_scene;
using Sha256 = db::cuda_active3_digest::Sha256;
using Clock = std::chrono::steady_clock;

class ExportError
  : public std::runtime_error
{
public:
  explicit ExportError (const std::string &message)
    : std::runtime_error (message)
  {
    //  nothing yet
  }
};

struct Options
{
  std::string input;
  std::string top;
  std::string output;
  std::uint32_t layer = 101;
  std::uint32_t datatype = 0;
  bool assert_merged = false;
};

double elapsed_seconds (const Clock::time_point &begin)
{
  return std::chrono::duration<double> (Clock::now () - begin).count ();
}

std::string errno_message (const std::string &operation)
{
  return operation + ": " + std::strerror (errno);
}

std::string hex_digest (const std::array<std::uint8_t, 32> &digest)
{
  std::ostringstream text;
  text << std::hex << std::setfill ('0');
  for (std::size_t i = 0; i < digest.size (); ++i) {
    text << std::setw (2) << unsigned (digest[i]);
  }
  return text.str ();
}

std::uint32_t parse_u32 (const std::string &text, const char *what)
{
  if (text.empty () || text[0] == '-') {
    throw ExportError (std::string ("invalid ") + what + ": " + text);
  }
  char *end = 0;
  errno = 0;
  const unsigned long long value = std::strtoull (text.c_str (), &end, 10);
  if (errno || ! end || *end ||
      value > std::numeric_limits<std::uint32_t>::max ()) {
    throw ExportError (std::string ("invalid ") + what + ": " + text);
  }
  return std::uint32_t (value);
}

bool take_value (
  const std::string &argument, const char *prefix, std::string &value)
{
  const std::size_t length = std::strlen (prefix);
  if (argument.compare (0, length, prefix) != 0) {
    return false;
  }
  value = argument.substr (length);
  return true;
}

Options parse_options (int argc, char **argv)
{
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument (argv[i]);
    std::string value;
    if (take_value (argument, "--input=", value)) {
      options.input = value;
    } else if (take_value (argument, "--top=", value)) {
      options.top = value;
    } else if (take_value (argument, "--output=", value)) {
      options.output = value;
    } else if (take_value (argument, "--layer=", value)) {
      options.layer = parse_u32 (value, "layer");
    } else if (take_value (argument, "--datatype=", value)) {
      options.datatype = parse_u32 (value, "datatype");
    } else if (argument == "--assert-merged") {
      options.assert_merged = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout
        << "usage: m1_width_space_host_scene_export "
        << "--input=CAPTURE.gds --top=TOP --output=SCENE.km1ws "
        << "[--layer=101] [--datatype=0] --assert-merged\n";
      std::exit (0);
    } else {
      throw ExportError ("unknown argument: " + argument);
    }
  }
  if (options.input.empty () || options.top.empty () ||
      options.output.empty ()) {
    throw ExportError ("input, top, and output are required");
  }
  if (! options.assert_merged) {
    throw ExportError (
      "refusing to serialize without explicit --assert-merged provenance");
  }
  if (options.input == options.output) {
    throw ExportError ("input and output paths alias textually");
  }
  return options;
}

std::array<std::uint8_t, 32> file_digest (const std::string &path)
{
  std::ifstream input (path.c_str (), std::ios::binary);
  if (! input) {
    throw ExportError ("cannot open input for SHA-256: " + path);
  }
  Sha256 sha;
  std::array<char, 1024 * 1024> buffer;
  while (input) {
    input.read (buffer.data (), std::streamsize (buffer.size ()));
    const std::streamsize count = input.gcount ();
    if (count > 0) {
      sha.update (buffer.data (), std::size_t (count));
    }
  }
  if (! input.eof ()) {
    throw ExportError ("failed while hashing input: " + path);
  }
  return sha.finish ();
}

unsigned int find_layer (
  const db::Layout &layout, std::uint32_t layer, std::uint32_t datatype)
{
  bool found = false;
  unsigned int result = 0;
  for (db::Layout::layer_iterator candidate = layout.begin_layers ();
       candidate != layout.end_layers (); ++candidate) {
    if ((*candidate).second->layer == int (layer) &&
        (*candidate).second->datatype == int (datatype)) {
      if (found) {
        throw ExportError ("source layer is ambiguous in the input layout");
      }
      found = true;
      result = (*candidate).first;
    }
  }
  if (! found) {
    throw ExportError (
      "source layer " + std::to_string (layer) + "/" +
      std::to_string (datatype) + " is absent");
  }
  return result;
}

class FileDescriptor
{
public:
  explicit FileDescriptor (int descriptor = -1)
    : m_descriptor (descriptor)
  {
    //  nothing yet
  }

  ~FileDescriptor ()
  {
    if (m_descriptor >= 0) {
      ::close (m_descriptor);
    }
  }

  FileDescriptor (const FileDescriptor &) = delete;
  FileDescriptor &operator= (const FileDescriptor &) = delete;

  int get () const
  {
    return m_descriptor;
  }

  int release ()
  {
    const int descriptor = m_descriptor;
    m_descriptor = -1;
    return descriptor;
  }

private:
  int m_descriptor;
};

void write_all (int descriptor, const void *data, std::size_t bytes)
{
  const std::uint8_t *cursor = static_cast<const std::uint8_t *> (data);
  while (bytes) {
    const std::size_t maximum =
      std::min<std::size_t> (
        bytes, std::size_t (std::numeric_limits<ssize_t>::max ()));
    const ssize_t written = ::write (descriptor, cursor, maximum);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw ExportError (errno_message ("write"));
    }
    if (! written) {
      throw ExportError ("write returned zero");
    }
    cursor += written;
    bytes -= std::size_t (written);
  }
}

void pwrite_all (
  int descriptor, const void *data, std::size_t bytes, std::uint64_t offset)
{
  const std::uint8_t *cursor = static_cast<const std::uint8_t *> (data);
  while (bytes) {
    const std::size_t maximum =
      std::min<std::size_t> (
        bytes, std::size_t (std::numeric_limits<ssize_t>::max ()));
    if (offset > std::uint64_t (std::numeric_limits<off_t>::max ())) {
      throw ExportError ("transport digest offset exceeds off_t");
    }
    const ssize_t written =
      ::pwrite (descriptor, cursor, maximum, off_t (offset));
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw ExportError (errno_message ("pwrite"));
    }
    if (! written) {
      throw ExportError ("pwrite returned zero");
    }
    cursor += written;
    bytes -= std::size_t (written);
    offset += std::uint64_t (written);
  }
}

class SceneSink
{
public:
  explicit SceneSink (int descriptor)
    : m_descriptor (descriptor), m_position (0)
  {
    //  nothing yet
  }

  void header (const void *data, std::size_t bytes)
  {
    write_all (m_descriptor, data, bytes);
    m_transport.update (data, bytes);
    m_position += bytes;
  }

  void payload (const void *data, std::size_t bytes)
  {
    write_all (m_descriptor, data, bytes);
    m_transport.update (data, bytes);
    m_semantic.update (data, bytes);
    m_position += bytes;
  }

  std::uint64_t position () const
  {
    return m_position;
  }

  std::array<std::uint8_t, 32> semantic_digest ()
  {
    return m_semantic.finish ();
  }

  std::array<std::uint8_t, 32> transport_digest ()
  {
    return m_transport.finish ();
  }

private:
  int m_descriptor;
  std::uint64_t m_position;
  Sha256 m_transport;
  Sha256 m_semantic;
};

void require_position (
  const SceneSink &sink, std::uint64_t expected, const char *section)
{
  if (sink.position () != expected) {
    throw ExportError (
      std::string ("internal ") + section + " offset mismatch");
  }
}

template <std::size_t N>
void emit_payload (
  SceneSink &sink, const std::array<std::uint8_t, N> &record)
{
  sink.payload (record.data (), record.size ());
}

std::array<std::uint8_t, format::kContextRecordBytes> encode_context (
  const db::CudaM1WidthSpaceContext &context)
{
  std::array<std::uint8_t, format::kContextRecordBytes> bytes = {};
  format::store_i64_le (bytes.data (), context.tx);
  format::store_i64_le (bytes.data () + 8, context.ty);
  format::store_u32_le (bytes.data () + 16, context.cell_id);
  format::store_u32_le (bytes.data () + 20, context.transform_code);
  return bytes;
}

std::array<std::uint8_t, format::kMetalContextRecordBytes>
encode_metal_context (
  std::uint32_t context, std::uint64_t polygon_offset,
  std::uint64_t edge_offset)
{
  std::array<std::uint8_t, format::kMetalContextRecordBytes> bytes = {};
  format::store_u32_le (bytes.data (), context);
  format::store_u64_le (bytes.data () + 4, polygon_offset);
  format::store_u64_le (bytes.data () + 12, edge_offset);
  return bytes;
}

std::array<std::uint8_t, format::kCellRecordBytes> encode_cell (
  const db::CudaM1WidthSpaceCell &cell)
{
  std::array<std::uint8_t, format::kCellRecordBytes> bytes = {};
  format::store_u64_le (bytes.data (), cell.source_cell_index);
  format::store_u64_le (bytes.data () + 8, cell.polygon_begin);
  format::store_u64_le (bytes.data () + 16, cell.edge_begin);
  format::store_u32_le (bytes.data () + 24, cell.polygon_count);
  format::store_u32_le (bytes.data () + 28, cell.edge_count);
  return bytes;
}

std::array<std::uint8_t, format::kPolygonRecordBytes> encode_polygon (
  const db::CudaM1WidthSpacePolygon &polygon)
{
  std::array<std::uint8_t, format::kPolygonRecordBytes> bytes = {};
  format::store_u64_le (bytes.data (), polygon.edge_begin);
  format::store_i64_le (bytes.data () + 8, polygon.left);
  format::store_i64_le (bytes.data () + 16, polygon.bottom);
  format::store_i64_le (bytes.data () + 24, polygon.right);
  format::store_i64_le (bytes.data () + 32, polygon.top);
  format::store_u32_le (bytes.data () + 40, polygon.polygon_id);
  format::store_u32_le (bytes.data () + 44, polygon.edge_count);
  return bytes;
}

std::array<std::uint8_t, format::kEdgeRecordBytes> encode_edge (
  const db::CudaM1WidthSpaceEdge &edge)
{
  std::array<std::uint8_t, format::kEdgeRecordBytes> bytes = {};
  format::store_i64_le (bytes.data (), edge.x1);
  format::store_i64_le (bytes.data () + 8, edge.y1);
  format::store_i64_le (bytes.data () + 16, edge.x2);
  format::store_i64_le (bytes.data () + 24, edge.y2);
  return bytes;
}

std::string parent_directory (const std::string &path)
{
  const std::string::size_type slash = path.find_last_of ('/');
  if (slash == std::string::npos) {
    return ".";
  }
  return slash ? path.substr (0, slash) : "/";
}

std::string base_name (const std::string &path)
{
  const std::string::size_type slash = path.find_last_of ('/');
  return slash == std::string::npos ? path : path.substr (slash + 1);
}

struct Publication
{
  std::string temporary;
  std::string output;
  bool published = false;

  ~Publication ()
  {
    if (! temporary.empty ()) {
      ::unlink (temporary.c_str ());
    }
    if (published) {
      ::unlink (output.c_str ());
    }
  }

  void complete ()
  {
    published = false;
    if (! temporary.empty ()) {
      ::unlink (temporary.c_str ());
      temporary.clear ();
    }
  }
};

void write_scene (
  const Options &options,
  const std::array<std::uint8_t, 32> &source_digest,
  const db::CudaM1WidthSpaceScene &scene,
  std::array<std::uint8_t, 32> &transport_digest)
{
  std::array<std::uint8_t, 32> recomputed;
  if (! db::cuda_m1_width_space_scene_digest (scene, recomputed) ||
      recomputed != scene.digest) {
    throw ExportError (
      "host scene failed canonical digest recomputation before export");
  }

  format::FileLayoutV1 layout;
  if (! format::compute_file_layout (
        scene.contexts.size (), scene.metal_contexts.size (),
        scene.cells.size (), scene.polygons.size (), scene.edges.size (),
        layout)) {
    throw ExportError ("scene file size arithmetic overflow");
  }
  if (layout.file_bytes >
      std::uint64_t (std::numeric_limits<off_t>::max ())) {
    throw ExportError ("scene file is too large for off_t");
  }

  format::FileHeaderV1 header = {};
  header.version = format::kFileVersion;
  header.header_bytes = std::uint32_t (format::kFileHeaderBytes);
  header.endian_tag = format::kEndianTag;
  header.flags = format::kRequiredFlags;
  header.layout = layout;
  header.context_count = scene.contexts.size ();
  header.metal_context_count = scene.metal_contexts.size ();
  header.cell_count = scene.cells.size ();
  header.polygon_count = scene.polygons.size ();
  header.edge_count = scene.edges.size ();
  header.scene_digest = scene.digest;
  header.source_digest = source_digest;
  header.source_layer = options.layer;
  header.source_datatype = options.datatype;

  format::SemanticHeaderV1 semantic = {};
  semantic.format_version = scene.format_version;
  semantic.dbu_per_micron = scene.dbu_per_micron;
  semantic.root_cell = scene.root_cell;
  semantic.reserved = scene.reserved;
  semantic.width_distance = scene.width_distance;
  semantic.spacing_distance = scene.spacing_distance;
  semantic.context_count = scene.contexts.size ();
  semantic.metal_context_count = scene.metal_contexts.size ();
  semantic.cell_count = scene.cells.size ();
  semantic.polygon_count = scene.polygons.size ();
  semantic.edge_count = scene.edges.size ();
  semantic.flat_polygon_count = scene.flat_polygon_count;
  semantic.flat_edge_count = scene.flat_edge_count;
  semantic.scene_left = scene.scene_left;
  semantic.scene_bottom = scene.scene_bottom;
  semantic.scene_right = scene.scene_right;
  semantic.scene_top = scene.scene_top;

  const std::string directory = parent_directory (options.output);
  const std::string base = base_name (options.output);
  if (base.empty ()) {
    throw ExportError ("output path has an empty basename");
  }
  Publication publication;
  publication.output = options.output;
  publication.temporary = directory + "/." + base + ".tmp.XXXXXX";
  std::vector<char> temporary (
    publication.temporary.begin (), publication.temporary.end ());
  temporary.push_back ('\0');
  const int raw_descriptor = ::mkstemp (temporary.data ());
  if (raw_descriptor < 0) {
    throw ExportError (errno_message ("mkstemp"));
  }
  publication.temporary.assign (temporary.data ());
  FileDescriptor descriptor (raw_descriptor);
  if (::fchmod (descriptor.get (), 0644) != 0) {
    throw ExportError (errno_message ("fchmod"));
  }

  SceneSink sink (descriptor.get ());
  const std::array<std::uint8_t, format::kFileHeaderBytes> header_bytes =
    format::encode_file_header (header);
  sink.header (header_bytes.data (), header_bytes.size ());
  require_position (sink, layout.payload_offset, "payload");

  emit_payload (sink, format::encode_semantic_header (semantic));
  require_position (sink, layout.contexts_offset, "context");
  for (std::size_t i = 0; i < scene.contexts.size (); ++i) {
    emit_payload (sink, encode_context (scene.contexts[i]));
  }

  require_position (sink, layout.metal_contexts_offset, "metal-context");
  for (std::size_t i = 0; i < scene.metal_contexts.size (); ++i) {
    emit_payload (
      sink, encode_metal_context (
        scene.metal_contexts[i],
        scene.context_polygon_offsets[i],
        scene.context_edge_offsets[i]));
  }

  require_position (sink, layout.cells_offset, "cell");
  for (std::size_t i = 0; i < scene.cells.size (); ++i) {
    emit_payload (sink, encode_cell (scene.cells[i]));
  }

  require_position (sink, layout.polygons_offset, "polygon");
  for (std::size_t i = 0; i < scene.polygons.size (); ++i) {
    emit_payload (sink, encode_polygon (scene.polygons[i]));
  }

  require_position (sink, layout.edges_offset, "edge");
  for (std::size_t i = 0; i < scene.edges.size (); ++i) {
    emit_payload (sink, encode_edge (scene.edges[i]));
  }

  require_position (sink, layout.file_bytes, "file");
  const std::array<std::uint8_t, 32> payload_digest =
    sink.semantic_digest ();
  if (payload_digest != scene.digest) {
    throw ExportError (
      "encoded canonical payload digest differs from host scene digest");
  }
  transport_digest = sink.transport_digest ();
  pwrite_all (
    descriptor.get (), transport_digest.data (), transport_digest.size (),
    format::kTransportDigestOffset);
  if (::fsync (descriptor.get ()) != 0) {
    throw ExportError (errno_message ("fsync temporary scene"));
  }
  if (::close (descriptor.release ()) != 0) {
    throw ExportError (errno_message ("close temporary scene"));
  }

  if (::link (
        publication.temporary.c_str (), publication.output.c_str ()) != 0) {
    throw ExportError (errno_message ("atomic no-clobber link"));
  }
  publication.published = true;

  FileDescriptor directory_descriptor (
    ::open (directory.c_str (), O_RDONLY | O_DIRECTORY));
  if (directory_descriptor.get () < 0 ||
      ::fsync (directory_descriptor.get ()) != 0) {
    throw ExportError (errno_message ("fsync output directory"));
  }
  publication.complete ();
}

db::CudaM1WidthSpaceScene build_scene (
  const Options &options,
  const std::array<std::uint8_t, 32> &expected_source_digest)
{
  const Clock::time_point load_begin = Clock::now ();
  db::Layout source_layout;
  {
    tl::InputStream input (options.input);
    db::GDS2Reader reader (input);
    reader.read (source_layout, db::LoadLayoutOptions ());
  }
  if (file_digest (options.input) != expected_source_digest) {
    throw ExportError ("input capture changed while it was being loaded");
  }
  std::cout
    << "loaded input cells=" << source_layout.cells ()
    << " dbu=" << source_layout.dbu ()
    << " seconds=" << elapsed_seconds (load_begin) << std::endl;

  const std::pair<bool, db::cell_index_type> top =
    source_layout.cell_by_name (options.top.c_str ());
  if (! top.first) {
    throw ExportError ("top cell is absent: " + options.top);
  }
  const unsigned int source_layer =
    find_layer (source_layout, options.layer, options.datatype);

  const Clock::time_point deep_begin = Clock::now ();
  db::DeepShapeStore store;
  // The input is already the exact merged capture.  Preserve its contours
  // instead of applying DeepShapeStore's normal DRC-oriented polygon
  // fragmentation (default vertex/area-ratio limits), which would inflate
  // the replay topology without changing its geometry.
  store.set_max_area_ratio (0.0);
  store.set_max_vertex_count (0);
  const db::RecursiveShapeIterator iterator (
    source_layout, source_layout.cell (top.second), source_layer);
  const db::DeepLayer metal1 =
    store.create_polygon_layer (iterator, 0.0, 0);
  std::cout
    << "built deep capture seconds=" << elapsed_seconds (deep_begin)
    << std::endl;

  db::CudaM1WidthSpaceBuildSpec spec;
  spec.inputs_are_merged = true;
  db::CudaM1WidthSpaceSceneLimits limits;
  db::CudaM1WidthSpaceScene scene;
  std::string decline;
  const Clock::time_point scene_begin = Clock::now ();
  if (! db::cuda_m1_width_space_build_scene (
        metal1, metal1, spec, limits, scene, &decline)) {
    throw ExportError ("host scene builder declined: " + decline);
  }
  std::cout
    << "built host scene"
    << " contexts=" << scene.contexts.size ()
    << " metal_contexts=" << scene.metal_contexts.size ()
    << " cells=" << scene.cells.size ()
    << " stored_polygons=" << scene.polygons.size ()
    << " stored_edges=" << scene.edges.size ()
    << " flat_polygons=" << scene.flat_polygon_count
    << " flat_edges=" << scene.flat_edge_count
    << " seconds=" << elapsed_seconds (scene_begin)
    << std::endl;
  return scene;
}

}  // namespace

int main (int argc, char **argv)
{
  try {
    const Options options = parse_options (argc, argv);
    const Clock::time_point total_begin = Clock::now ();
    const Clock::time_point source_begin = Clock::now ();
    const std::array<std::uint8_t, 32> source_sha =
      file_digest (options.input);
    std::cout
      << "source_sha256=" << hex_digest (source_sha)
      << " seconds=" << elapsed_seconds (source_begin) << std::endl;

    const db::CudaM1WidthSpaceScene scene =
      build_scene (options, source_sha);
    std::array<std::uint8_t, 32> transport_sha;
    const Clock::time_point write_begin = Clock::now ();
    write_scene (options, source_sha, scene, transport_sha);
    std::cout
      << "KM1WSCN1 output=" << options.output
      << " scene_sha256=" << hex_digest (scene.digest)
      << " transport_sha256=" << hex_digest (transport_sha)
      << " write_seconds=" << elapsed_seconds (write_begin)
      << " total_seconds=" << elapsed_seconds (total_begin)
      << std::endl;
    return 0;
  } catch (const tl::Exception &ex) {
    std::cerr << "m1 host-scene export failed: " << ex.msg () << std::endl;
  } catch (const std::exception &ex) {
    std::cerr << "m1 host-scene export failed: " << ex.what () << std::endl;
  } catch (...) {
    std::cerr << "m1 host-scene export failed: unknown exception" << std::endl;
  }
  return 1;
}
