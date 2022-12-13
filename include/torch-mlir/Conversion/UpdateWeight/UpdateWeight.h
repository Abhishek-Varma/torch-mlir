//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_UPDATEWEIGHT_UPDATEWEIGHT_H
#define TORCHMLIR_CONVERSION_UPDATEWEIGHT_UPDATEWEIGHT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
namespace torch {

std::unique_ptr<OperationPass<ModuleOp>> createUpdateWeightPass();

std::unique_ptr<OperationPass<ModuleOp>> createMarkWeightPass();
}
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_UPDATEWEIGHT_UPDATEWEIGHT_H
