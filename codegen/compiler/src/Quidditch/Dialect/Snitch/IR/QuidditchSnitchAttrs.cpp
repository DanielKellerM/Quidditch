#include "QuidditchSnitchAttrs.h"

using namespace mlir;
using namespace quidditch::Snitch;

//===----------------------------------------------------------------------===//
// LoweringConfigAttr::LoweringConfigAttrInterface
//===----------------------------------------------------------------------===//

SmallVector<int64_t> LoweringConfigAttr::getWorkgroupTileSizes() const {
  return llvm::to_vector(getWorkgroupTiles());
}

// Gate consulted by TileAndDistributeToWorkgroupsUsingForallOp: without this the
// attr inherits the interface default (false) and the pass never tiles to
// workgroups (grid stays {1,1,1}). Non-empty leading workgroup_tiles => tile.
bool LoweringConfigAttr::hasWorkgroupTilingLevel() const {
  return !getWorkgroupTileSizes().empty();
}
