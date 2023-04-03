// RUN: torch-mlir-opt <%s -convert-torch-to-tmtensor -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @torch.aten.sort(
// CHECK-SAME:         %[[INPUT_ARG:.*]]: !torch.vtensor<[3,4],f32>)
// CHECK:           %[[INPUT_TENSOR:.*]] = torch_c.to_builtin_tensor %[[INPUT_ARG]] : !torch.vtensor<[3,4],f32> -> tensor<3x4xf32>
// CHECK:           %[[INDICES_TENSOR:.*]] = tensor.generate  {
// CHECK:                   ^bb0(%[[i:.*]]: index, %[[j:.*]]: index):
// CHECK:                       %[[j_index:.*]] = arith.index_cast %[[j]] : index to i64
// CHECK:                       tensor.yield %[[j_index]] : i64
// CHECK:           } : tensor<3x4xi64>
// CHECK:           %[[SORT:.*]]:2 = tm_tensor.sort dimension(1) outs(%[[INPUT_TENSOR]], %[[INDICES_TENSOR]] : tensor<3x4xf32>, tensor<3x4xi64>) {
// CHECK:                   ^bb0(%[[INP1:.*]]: f32, %[[INP2:.*]]: f32, %[[IND1:.*]]: i64, %[[IND2:.*]]: i64):
// CHECK:                       %[[PREDICATE:.*]] = arith.cmpf ole, %[[INP1]], %[[INP2]] : f32
// CHECK:                       tm_tensor.yield %4 : i1
// CHECK:           } -> tensor<3x4xf32>, tensor<3x4xi64>
// CHECK:           %[[OUTPUT_SORTED_TENSOR:.*]] = torch_c.from_builtin_tensor %[[SORT]]#0 : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           %[[OUTPUT_INDICES_TENSOR:.*]] = torch_c.from_builtin_tensor %[[SORT]]#1 : tensor<3x4xi64> -> !torch.vtensor<[3,4],si64>
// CHECK:           return %[[OUTPUT_SORTED_TENSOR]], %[[OUTPUT_INDICES_TENSOR]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>
// CHECK:        }
func.func @torch.aten.sort(%arg0: !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>) {
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %values, %indices = torch.aten.sort %arg0, %int-1, %false : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>
    return %values, %indices : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],si64>
}