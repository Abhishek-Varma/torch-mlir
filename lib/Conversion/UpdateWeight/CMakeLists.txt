add_mlir_conversion_library(TorchMLIRUpdateWeight
  UpdateWeight.cpp
  MarkWeight.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/UpdateWeight

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRMathDialect
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRUpdateWeight)