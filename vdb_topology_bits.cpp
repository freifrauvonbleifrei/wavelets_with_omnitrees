// vdb_topology_bits.cpp
//
// Prints per-level node counts and information-theoretic topology bits
// for each grid in a .vdb file. Topology bits = active masks only,
// no pointers, no padding, no value storage.
//
// Build (adjust include/lib paths as needed):
// g++ -std=c++17 -O2 vdb_topology_bits.cpp \
//   -I${SPACK_PATH}/...openvdb.../include \
//   -L${SPACK_PATH}/...openvdb.../lib \
//   -Wl,-rpath,${SPACK_PATH}/...openvdb.../lib \
//   -lopenvdb -ltbb \
//   -o vdb_topology_bits
//
// Usage:
//   ./vdb_topology_bits file.vdb [gridname]
//   ./vdb_topology_bits file.vdb            # processes all grids
//
// Output (one JSON object per line, for easy subprocess parsing):
//   {"grid":"density","level":0,"type":"Leaf8","nodes":1234,"mask_bits_per_node":512,"total_bits":631808}
//   {"grid":"density","level":1,"type":"Internal16","nodes":56,"mask_bits_per_node":8192,"total_bits":458752}
//   ...
//   {"grid":"density","summary":true,"total_topology_bits":1234567}

#include <iostream>
#include <openvdb/openvdb.h>
#include <openvdb/tree/NodeManager.h>
#include <string>
#include <vector>

// Mask bits for each node type in the default Tree<float,5,4,3> config:
//   Leaf(8^3):        512-bit  value mask
//   Internal(16^3):  4096-bit  child mask + 4096-bit value mask = 8192 bits
//   Internal(32^3): 32768-bit  child mask + 32768-bit value mask = 65536 bits
//   Root:            variable  (sparse hash map — excluded from count)
//
// For a general Tree<V, N2, N1, N0>:
//   Leaf dim      = 2^N0,  mask bits = (2^N0)^3
//   Internal1 dim = 2^N1,  mask bits = 2 * (2^N1)^3
//   Internal2 dim = 2^N2,  mask bits = 2 * (2^N2)^3

template <typename GridType>
void processGrid(const std::string &name, typename GridType::Ptr grid) {
  using TreeType = typename GridType::TreeType;

  // nodeCount() returns a vector indexed 0=leaf ... depth-1=just-below-root
  // (root itself is excluded from the vector in OpenVDB's convention)
  std::vector<openvdb::Index64> counts = grid->tree().nodeCount();
  // counts[0] = leaf count, counts[1] = first internal, counts[2] = second
  // internal, ...

  // Derive mask sizes from the tree's compile-time log2dims.
  // Default Tree<V,5,4,3>: leaf log2dim=3, internal log2dims=4,5
  // We compute them generically from the actual node types.

  using LeafT = typename TreeType::LeafNodeType;
  using Internal1 =
      typename TreeType::RootNodeType::ChildNodeType::ChildNodeType;
  using Internal2 = typename TreeType::RootNodeType::ChildNodeType;

  constexpr openvdb::Index64 leafMaskBits =
      static_cast<openvdb::Index64>(LeafT::NUM_VOXELS); // value mask only

  constexpr openvdb::Index64 int1MaskBits =
      2 * static_cast<openvdb::Index64>(
              Internal1::NUM_VALUES); // child + value mask

  constexpr openvdb::Index64 int2MaskBits =
      2 * static_cast<openvdb::Index64>(
              Internal2::NUM_VALUES); // child + value mask

  // Level names and mask sizes, leaf-first to match counts[] ordering
  struct LevelInfo {
    std::string type;
    openvdb::Index64 maskBitsPerNode;
  };

  std::vector<LevelInfo> levels;
  levels.push_back({"Leaf_" + std::to_string(LeafT::DIM), leafMaskBits});
  if (counts.size() >= 2)
    levels.push_back(
        {"Internal_" + std::to_string(Internal1::DIM), int1MaskBits});
  if (counts.size() >= 3)
    levels.push_back(
        {"Internal_" + std::to_string(Internal2::DIM), int2MaskBits});

  openvdb::Index64 totalBits = 0;

  for (size_t i = 0; i < counts.size() && i < levels.size(); ++i) {
    openvdb::Index64 levelBits = counts[i] * levels[i].maskBitsPerNode;
    totalBits += levelBits;

    std::cout << "{" << "\"grid\":\"" << name << "\"," << "\"level\":" << i
              << "," << "\"type\":\"" << levels[i].type << "\","
              << "\"nodes\":" << counts[i] << ","
              << "\"mask_bits_per_node\":" << levels[i].maskBitsPerNode << ","
              << "\"total_bits\":" << levelBits << "}\n";
  }

  std::cout << "{" << "\"grid\":\"" << name << "\"," << "\"summary\":true,"
            << "\"total_topology_bits\":" << totalBits << ","
            << "\"total_topology_bytes\":" << (totalBits / 8) << "}\n";

  // ── Value storage ────────────────────────────────────────────────────────
  using ValueType = typename GridType::ValueType;
  constexpr openvdb::Index64 bitsPerValue = sizeof(ValueType) * 8;

  // Dense: full leaf buffer (8^3 values per leaf) regardless of active count,
  // plus every child slot at internal nodes (used for tiles, active or not).
  constexpr openvdb::Index64 leafDenseBits =
      static_cast<openvdb::Index64>(LeafT::NUM_VOXELS) * bitsPerValue;
  constexpr openvdb::Index64 int1DenseBits =
      static_cast<openvdb::Index64>(Internal1::NUM_VALUES) * bitsPerValue;
  constexpr openvdb::Index64 int2DenseBits =
      static_cast<openvdb::Index64>(Internal2::NUM_VALUES) * bitsPerValue;

  openvdb::Index64 denseValueBits = 0;
  if (counts.size() > 0)
    denseValueBits += counts[0] * leafDenseBits;
  if (counts.size() > 1)
    denseValueBits += counts[1] * int1DenseBits;
  if (counts.size() > 2)
    denseValueBits += counts[2] * int2DenseBits;

  // Active-only: only voxels/tiles that are actually active.
  // activeLeafVoxelCount: active voxels across all leaf nodes.
  // activeTileCount:      active tiles at all internal + root levels
  //                       (regions uniform enough to not be subdivided).
  const openvdb::Index64 activeLeafVoxels = grid->tree().activeLeafVoxelCount();
  const openvdb::Index64 activeTiles = grid->tree().activeTileCount();
  const openvdb::Index64 leafCount = (counts.size() > 0) ? counts[0] : 0;
  const openvdb::Index64 totalLeafVoxels =
      leafCount * static_cast<openvdb::Index64>(LeafT::NUM_VOXELS);
  const openvdb::Index64 inactiveLeafVoxels = totalLeafVoxels - activeLeafVoxels;
  const openvdb::Index64 totalCoefficients = totalLeafVoxels + activeTiles;

  const openvdb::Index64 activeValueBits =
      (activeLeafVoxels + activeTiles) * bitsPerValue;

  std::cout << "{" << "\"grid\":\"" << name << "\","
            << "\"value_summary\":true,"
            << "\"bits_per_value\":" << bitsPerValue << ","
            << "\"leaf_count\":" << leafCount << ","
            << "\"active_leaf_voxels\":" << activeLeafVoxels << ","
            << "\"inactive_leaf_voxels\":" << inactiveLeafVoxels << ","
            << "\"active_tiles\":" << activeTiles << ","
            << "\"total_leaf_coefficients\":" << totalLeafVoxels << ","
            << "\"total_coefficients\":" << totalCoefficients << ","
            << "\"dense_value_bits\":" << denseValueBits << ","
            << "\"dense_value_bytes\":" << (denseValueBits / 8) << ","
            << "\"active_value_bits\":" << activeValueBits << ","
            << "\"active_value_bytes\":" << (activeValueBits / 8) << "}\n";
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " file.vdb [gridname]\n";
    return 1;
  }

  const std::string filename = argv[1];
  const std::string filterName = (argc >= 3) ? argv[2] : "";

  openvdb::initialize();

  openvdb::io::File file(filename);
  try {
    file.open();
  } catch (const openvdb::IoError &e) {
    std::cerr << "Error opening " << filename << ": " << e.what() << "\n";
    return 1;
  }

  for (auto nameIter = file.beginName(); nameIter != file.endName();
       ++nameIter) {
    const std::string &gridName = nameIter.gridName();
    if (!filterName.empty() && gridName != filterName)
      continue;

    openvdb::GridBase::Ptr baseGrid = file.readGrid(gridName);

    // Dispatch on grid type — extend as needed for other value types
    if (auto g =  openvdb::gridPtrCast<openvdb::BoolGrid>(baseGrid))
      processGrid<openvdb::BoolGrid>(gridName, g);
    else
      std::cerr << "Skipping grid \"" << gridName
                << "\": unsupported value type\n";
  }

  file.close();
  return 0;
}