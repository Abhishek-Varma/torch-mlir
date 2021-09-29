//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_PASSES_H
#define TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

#include <memory>

namespace mlir {
namespace torch {
namespace TorchConversion {

/// Creates a pipeline that lowers the object graph IR that is given by a
/// TorchScript jit.ScriptModule into the form expected by
/// torch-verify-linalg-on-tensors-verify-backend-contract.
void createTorchScriptModuleToLinalgOnTensorsBackendPipeline(
    OpPassManager &pm,
    const torch::Torch::TorchLoweringPipelineOptions &options);

/// Creates a pipeline that lowers the object graph IR that is given by a
/// TorchScript jit.ScriptFunction into the form expected by
/// torch-verify-linalg-on-tensors-verify-backend-contract.
void createTorchScriptFunctionToLinalgOnTensorsBackendPipeline(
    OpPassManager &pm,
    const torch::Torch::TorchLoweringPipelineOptions &options);

std::unique_ptr<OperationPass<ModuleOp>>
createVerifyInvariantsBeforeBackendLoweringPass();

std::unique_ptr<OperationPass<ModuleOp>> createFuncBackendTypeConversionPass();

std::unique_ptr<OperationPass<FuncOp>>
createFinalizingBackendTypeConversionPass();

std::unique_ptr<OperationPass<ModuleOp>>
createVerifyLinalgOnTensorsBackendContractPass();

} // namespace TorchConversion

/// Registers all Torch transformation passes.
void registerTorchConversionPasses();

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_PASSES_H
