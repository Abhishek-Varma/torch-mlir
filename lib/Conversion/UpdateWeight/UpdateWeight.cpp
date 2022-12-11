//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/UpdateWeight/UpdateWeight.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::torch;

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
/// This pass extracts the `tensor.literal` ops from the given torchscript IR and
/// updates the corresponding weight tensors (`arith.const`) ops in a Linalg IR
/// pointed to by an environment variable.
class UpdateWeight : public UpdateWeightBase<UpdateWeight> {
public:
  void runOnOperation() override {
    // Fetch the ModuleOp from the Torchscript IR.
    ModuleOp newResourceModuleOp = getOperation();
    MLIRContext *context = newResourceModuleOp->getContext();
    // `newResourceMap` contains mapping of new weight/resource's ID to their value. 
    DenseMap<unsigned, ElementsAttr> newResourceMap;
    // Here we get hold of the `torch.nn_module` inside the ModuleOp and iterate over
    // the `torch.slot` ops within it. Only those `torch.slot` ops with the StrAttr
    // attached like `_param_constant<num>` is used to add entry to `newResourceMap`.
    newResourceModuleOp.walk([&](torch::Torch::NnModuleOp nnModuleOp) {
      nnModuleOp.walk([&](torch::Torch::SlotOp slotOp) {
        std::string slotName = slotOp.getName().str();
        if (slotName.rfind("_param_constant", 0) == 0) {
          auto tensorLiteral = slotOp.getValue().getDefiningOp<torch::Torch::NonValueTensorLiteralOp>();
          ElementsAttr resourceValue = tensorLiteral.getValue();
          unsigned resourceId = std::stoi(slotName.substr(15));
          newResourceMap.insert({resourceId, resourceValue});
        }
      });
    });

    // We now fetch the Linalg IR file to update weights into.
    StringRef initialLinalgIRFilePath = std::getenv("INITIAL_LINALG_IR");
    llvm::SourceMgr sourceMgr;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(initialLinalgIRFilePath);
    if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
      return;
    }
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context);
    ModuleOp initialLinalgIRModuleOp = *module;
    // In the Linalg IR we get hold of those `arith.const` ops with `resource_id`
    // attribute attached and update their constant values by using `newResourceMap`.
    initialLinalgIRModuleOp.walk([&](func::FuncOp funcOp){
      funcOp.walk([&](arith::ConstantOp arithConstantOp) {
        auto resourceIdAttr = arithConstantOp->getAttr("resource_id");
        if (resourceIdAttr) {
          unsigned resourceId = resourceIdAttr.cast<IntegerAttr>().getInt();
          // If `newResourceMap` has a record with `resouceId` as key, we extract the
          // new value of the weight to be updated and replace the constant value of
          // the `arith.const` op.
          if (newResourceMap.count(resourceId) == 1) {
            arithConstantOp->setAttr("value", newResourceMap[resourceId]);
          }
        }
      });
    });

    // Finally we replace ModuleOp's region of the Torchscript IR with that of the LinalgIR's.
    newResourceModuleOp->getRegion(0).takeBody(initialLinalgIRModuleOp->getRegion(0));
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::torch::createUpdateWeightPass() {
  return std::make_unique<UpdateWeight>();
}
