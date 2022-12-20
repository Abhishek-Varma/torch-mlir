//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOMLPROGRAM_TORCHTOMLPROGRAM_H
#define TORCHMLIR_CONVERSION_TORCHTOMLPROGRAM_TORCHTOMLPROGRAM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace torch {
std::unique_ptr<OperationPass<ModuleOp>> createConvertTorchToMLProgramPass();
}
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOMLPROGRAM_TORCHTOMLPROGRAM_H