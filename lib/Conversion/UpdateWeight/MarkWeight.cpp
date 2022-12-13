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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"


using namespace mlir;
using namespace mlir::torch;

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
/// This pass marks the `tensor.literal` ops with `resource_id` from the given
/// torchscript IR.
class MarkWeight : public MarkWeightBase<MarkWeight> {
public:
  void runOnOperation() override {
    // Fetch the ModuleOp from the Torchscript IR.
    ModuleOp newResourceModuleOp = getOperation();
    // Here we get hold of the `torch.nn_module` inside the ModuleOp and iterate over
    // the `torch.slot` ops within it. Only those `torch.slot` ops with the StrAttr
    // attached like `_param_constant<num>` is used to mark the corresponding
    // `tensor.literal` op.
    newResourceModuleOp.walk([&](torch::Torch::NnModuleOp nnModuleOp) {
      nnModuleOp.walk([&](torch::Torch::SlotOp slotOp) {
        std::string slotName = slotOp.getName().str();
        if (slotName.rfind("_param_constant", 0) == 0) {
          auto tensorLiteral = slotOp.getValue().getDefiningOp<torch::Torch::NonValueTensorLiteralOp>();
          int64_t resourceId = std::stoi(slotName.substr(15));
          OpBuilder builder(tensorLiteral);
          tensorLiteral->setAttr("resource_id", builder.getI64IntegerAttr(resourceId));
        }
      });
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::torch::createMarkWeightPass() {
  return std::make_unique<MarkWeight>();
}
