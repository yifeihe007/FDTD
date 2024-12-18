# RUN: %PYTHON %s | FileCheck %s

import ctypes
import sys
import os
import subprocess
import math
import time
from cavity_check import check_fields
# import matplotlib
# matplotlib.use('TkAgg')






from mlir.passmanager import PassManager
from mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from mlir.dialects import scf, pdl, func, arith, linalg
from mlir.dialects.transform import (
    get_parent_op,
    apply_patterns_canonicalization,
    apply_cse,
    any_op_t,
    ApplyCanonicalizationPatternsOp,
    apply_registered_pass
)
from mlir.dialects.transform.structured import (
    structured_match, 
    ApplyTilingCanonicalizationPatternsOp,
    VectorizeOp,
    ApplyFoldUnitExtentDimsViaReshapesPatternsOp,
    MatchOp,
    HoistRedundantVectorTransfersOp
)
from mlir.dialects.transform.tensor import ( 
    ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp,
)
from mlir.dialects.transform.vector import ( 
    ApplyLowerTransferPatternsOp,
    ApplyLowerTransposePatternsOp,
    ApplyLowerShapeCastPatternsOp,
    ApplyTransferToScfPatternsOp,
    ApplyCastAwayVectorLeadingOneDimPatternsOp,
    ApplyRankReducingSubviewPatternsOp,
    ApplyTransferPermutationPatternsOp
)
from mlir.dialects.transform.memref import ( 
    ApplyFoldMemrefAliasOpsPatternsOp,
    ApplyAllocToAllocaOp,
)
from mlir.dialects.transform.loop import loop_unroll
from mlir.dialects.transform.extras import named_sequence, apply_patterns
from mlir.extras import types as T
from mlir.dialects.builtin import module, ModuleOp









from mlir.execution_engine import *
from mlir import runtime as rt
from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import func
from mlir.dialects import linalg
from mlir.dialects import tensor
from mlir.dialects import arith
from mlir.dialects import scf
from mlir.dialects import transform, pdl, memref
from mlir.dialects.transform.extras import named_sequence, apply_patterns
from mlir.dialects.transform import loop, bufferization, structured
from mlir.dialects.bufferization import LayoutMapOption





from mlir.dialects.linalg.opdsl.lang import *

T1 = TV.T1
T2 = TV.T2

NX = 1024
NY = 1024
NZ = 1024

EPS0 = 8.8541878e-12;         # Permittivity of vacuum
MU0  = 4e-7 * np.pi;          # Permeability of vacuum
c0   = 299792458.0

# Parameter initiation
Lx = .05; Ly = .04; Lz = .03; # Cavity dimensions in meters
Nt = 1;                    # Number of time steps


DX = Lx / NX; DY = Ly / NY; DZ = Lz / NZ

DT = 0.5 * (DX / (c0 * np.sqrt(3)))


LLVM_PATH = "/mimer/NOBACKUP/groups/snic2022-22-1035/yifei/fdtd/20241104/llvm-project"
FILE_PATH = "/mimer/NOBACKUP/groups/snic2022-22-1035/yifei/fdtd/20241104/test/"

print_boiler = """
func.func @debug_print(%arg0: tensor<?x?x?xf32>) {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
%dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
%dim_2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
vector.print str "Debuging!"
scf.for %arg1 = %c0 to %dim step %c1 {
    scf.for %arg2 = %c0 to %dim_1 step %c1 {
    scf.for %arg3 = %c0 to %dim_2 step %c1 {
        %extracted = tensor.extract %arg0[%arg1, %arg2, %arg3] : tensor<?x?x?xf32>
        vector.print %extracted : f32
    }
    }
}
return
}
"""




def print_to_file(module, filename):
# Open the file in write mode ('w'). Use 'a' to append to an existing file.
    with open(filename, 'w') as file:
        # Use the print function, specifying the file parameter
        #print(module)
        print(module, file=file)


def test_fdtd_program(Nx: int, Ny: int, Nz: int):

    module = Module.create()
    f16 = F16Type.get()
    f32 = F32Type.get()
    f64 = F64Type.get()
    dataType = F32Type.get()
    i8 = IntegerType.get_signless(8)
    i16 = IntegerType.get_signless(16)
    i32 = IntegerType.get_signless(32)

    with InsertionPoint(module.body):
        
        
        @func.FuncOp.from_py_func(
            RankedTensorType.get((Nx+1, Ny + 1+1, Nz + 1+1), dataType),
                                    RankedTensorType.get((Nx + 1+1, Ny+1, Nz + 1+1), dataType),
                                    RankedTensorType.get((Nx + 1+1, Ny + 1+1, Nz+1), dataType),
                                    RankedTensorType.get((Nx + 1, Ny, Nz), dataType),
                                    RankedTensorType.get((Nx, Ny + 1, Nz), dataType),
                                    RankedTensorType.get((Nx, Ny, Nz + 1), dataType),
                                    
                        # RankedTensorType.get((ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size()), f32),
                        # RankedTensorType.get((ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size()), f32),
                        # RankedTensorType.get((ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size()), f32),
                        # RankedTensorType.get((ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size()), f32),
                        # RankedTensorType.get((ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size()), f32),
                        # RankedTensorType.get((ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size()), f32),

                                    name = "fdtd_" + str(Nx) +
                                "_" + str(Ny) + "_" + str(Nz))        
        def test_H(Ex, Ey, Ez, Hx, Hy, Hz):
            tensor_type_target = RankedTensorType.get([ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size(), ShapedType.get_dynamic_size()], F32Type.get())
            # casted_tensor1 = tensor.CastOp(tensor_type_target, Hx).result
            # func.CallOp([], "debug_print", [casted_tensor1]) 
            offsets_0 = [0, 0, 0]
            offsets_x = [1, 0, 0]
            offsets_y = [0, 1, 0]
            offsets_z = [0, 0, 1]

            sizes = [Nx, Ny, Nz]
            strides = [1, 1, 1]

            offsetsAttr = DenseI64ArrayAttr.get(offsets_0)
            offsetsXAttr = DenseI64ArrayAttr.get(offsets_x)
            offsetsYAttr = DenseI64ArrayAttr.get(offsets_y)
            offsetsZAttr = DenseI64ArrayAttr.get(offsets_z)


            sizesAttr = DenseI64ArrayAttr.get(sizes)
            stridesAttr = DenseI64ArrayAttr.get(strides)
            result_type = RankedTensorType.get(sizes, dataType)


            real_zero = arith.ConstantOp(dataType, 0.0)
            Coef_H = arith.ConstantOp(dataType, DT / MU0)
            Coef_E = arith.ConstantOp(dataType, DT / EPS0)
            # print("CE", DT / EPS0)

            Dx = arith.ConstantOp(dataType, DX)
            Dy = arith.ConstantOp(dataType, DY)
            Dz = arith.ConstantOp(dataType, DZ)
            
            TIME_BLOCK = 10                 # Number of time steps in each temporal block
            total_timesteps = 8192
            lower_bound = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), 0)).result
            upper_bound = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), total_timesteps)).result
            step = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), TIME_BLOCK)).result

    
            # Create the `scf.for` loop
            # loop = scf.ForOp(lower_bound, upper_bound, step)
    
            # # Define the loop body
            # with InsertionPoint(loop.body):



#   for i in range(0,Nx):
#         for j in range(0,Ny):
#             for k in range(0,Nz):
#                  Hx[i,j,k] += (dt / mu0)*((Ey[i,j,k+1]-Ey[i,j,k])/Dz - (Ez[i,j+1,k]-Ez[i,j,k])/Dy);
#                  Hy[i,j,k] += (dt / mu0)*((Ez[i+1,j,k]-Ez[i,j,k])/Dx - (Ex[i,j,k+1]-Ex[i,j,k])/Dz);
#                  Hz[i,j,k] += (dt / mu0)*((Ex[i,j+1,k]-Ex[i,j,k])/Dy - (Ey[i+1,j,k]-Ey[i,j,k])/Dx);

            # Create the extract_slice operation
            Ex0 = tensor.ExtractSliceOp(
                result_type,
                Ex,
                [],
                [],
                [],
                offsetsAttr,
                sizesAttr,
                stridesAttr,
            )

            Ex_y = tensor.ExtractSliceOp(
                result_type,
                Ex,
                [],
                [],
                [],
                offsetsYAttr,
                sizesAttr,
                stridesAttr,
            )
            
            Ex_z = tensor.ExtractSliceOp(
                result_type,
                Ex,
                [],
                [],
                [],
                offsetsZAttr,
                sizesAttr,
                stridesAttr,
            )
                            
            Ey0 = tensor.ExtractSliceOp(
                result_type,
                Ey,
                [],
                [],
                [],
                offsetsAttr,
                sizesAttr,
                stridesAttr,
            )

            Ey_z = tensor.ExtractSliceOp(
                result_type,
                Ey,
                [],
                [],
                [],
                offsetsZAttr,
                sizesAttr,
                stridesAttr,
            )
            
            Ey_x = tensor.ExtractSliceOp(
                result_type,
                Ey,
                [],
                [],
                [],
                offsetsXAttr,
                sizesAttr,
                stridesAttr,
            )
            
            Ez0 = tensor.ExtractSliceOp(
                result_type,
                Ez,
                [],
                [],
                [],
                offsetsAttr,
                sizesAttr,
                stridesAttr,
            )

            Ez_x = tensor.ExtractSliceOp(
                result_type,
                Ez,
                [],
                [],
                [],
                offsetsXAttr,
                sizesAttr,
                stridesAttr,
            )

            Ez_y = tensor.ExtractSliceOp(
                result_type,
                Ez,
                [],
                [],
                [],
                offsetsYAttr,
                sizesAttr,
                stridesAttr,
            )

            Hz0 = tensor.ExtractSliceOp(
                result_type,
                Hz,
                [],
                [],
                [],
                offsetsAttr,
                sizesAttr,
                stridesAttr,
            )
            
            Hy0 = tensor.ExtractSliceOp(
                result_type,
                Hy,
                [],
                [],
                [],
                offsetsAttr,
                sizesAttr,
                stridesAttr,
            )
            
            Hx0 = tensor.ExtractSliceOp(
                result_type,
                Hx,
                [],
                [],
                [],
                offsetsAttr,
                sizesAttr,
                stridesAttr,
            )

#   for i in range(0,Nx):
#         for j in range(0,Ny):
#             for k in range(0,Nz):
#                  Hx[i,j,k] += (dt / mu0)*((Ey[i,j,k+1]-Ey[i,j,k])/Dz - (Ez[i,j+1,k]-Ez[i,j,k])/Dy);
#                  Hy[i,j,k] += (dt / mu0)*((Ez[i+1,j,k]-Ez[i,j,k])/Dx - (Ex[i,j,k+1]-Ex[i,j,k])/Dz);
#                  Hz[i,j,k] += (dt / mu0)*((Ex[i,j+1,k]-Ex[i,j,k])/Dy - (Ey[i+1,j,k]-Ey[i,j,k])/Dx);


            Hz1 = linalg.curl_step(Ex_y, Ex0, Ey_x, Ey0, Coef_H,
                                        Dy, Dx, outs=[Hz0])
            # casted_tensor1 = tensor.CastOp(tensor_type_target, Hz).result
            # func.CallOp([], "debug_print", [casted_tensor1]) 
     
            Hy1 = linalg.curl_step(Ez_x, Ez0, Ex_z, Ex0, Coef_H,
                                        Dx, Dz, outs=[Hy0])

            Hx1 = linalg.curl_step(Ey_z, Ey0, Ez_y, Ez0, Coef_H,
                                        Dz, Dy, outs=[Hx0])
   
     


            
            # Create the extract_slice operation
            Hz2 = tensor.InsertSliceOp(
                Hz1,
                Hz,
                [],
                [],
                [],
                offsetsAttr,
                sizesAttr,
                stridesAttr,
            )
            
            Hy2 = tensor.InsertSliceOp(
                Hy1,
                Hy,
                [],
                [],
                [],
                offsetsAttr,
                sizesAttr,
                stridesAttr,
            )
            
            Hx2 = tensor.InsertSliceOp(
                Hx1,
                Hx,
                [],
                [],
                [],
                offsetsAttr,
                sizesAttr,
                stridesAttr,
            )

# def H_edge():
#     for j in range(Ny):
#       for k in range(Nz):
#           Hx[Nx,j,k] += (dt / mu0)*((Ey[Nx,j,k+1]-Ey[Nx,j,k])/Dz - (Ez[Nx,j+1,k]-Ez[Nx,j,k])/Dy);
#     for i in range(Nx):
#       for k in range(Nz):
#           Hy[i,Ny,k] += (dt / mu0)*((Ez[i+1,Ny,k]-Ez[i,Ny,k])/Dx - (Ex[i,Ny,k+1]-Ex[i,Ny,k])/Dz);
#     for i in range(Nx):
#       for j in range(Ny):
#           Hz[i,j,Nz] += (dt / mu0)*((Ex[i,j+1,Nz]-Ex[i,j,Nz])/Dy - (Ey[i+1,j,Nz]-Ey[i,j,Nz])/Dx);

                # Create the extract_slice operation
                
                
            offsets_0_x = [Nx, 0, 0]
            offsets_0_y = [0, Ny, 0]
            offsets_0_z = [0, 0, Nz]
            
            offsets_y_x = [Nx, 1, 0]
            offsets_z_x = [Nx, 0, 1]

            offsets_x_y = [1, Ny, 0]
            offsets_z_y = [0, Ny, 1]

            offsets_x_z = [1, 0, Nz]
            offsets_y_z = [0, 1, Nz]

            sizes_x = [1, Ny, Nz]
            sizes_y = [Nx, 1, Nz]
            sizes_z = [Nx, Ny, 1]

            strides = [1, 1, 1]

            offsets0XAttr = DenseI64ArrayAttr.get(offsets_0_x)
            offsets0YAttr = DenseI64ArrayAttr.get(offsets_0_y)
            offsets0ZAttr = DenseI64ArrayAttr.get(offsets_0_z)
            offsetsYXAttr = DenseI64ArrayAttr.get(offsets_y_x)
            offsetsZXAttr = DenseI64ArrayAttr.get(offsets_z_x)
            offsetsXYAttr = DenseI64ArrayAttr.get(offsets_x_y)
            offsetsZYAttr = DenseI64ArrayAttr.get(offsets_z_y)
            offsetsXZAttr = DenseI64ArrayAttr.get(offsets_x_z)
            offsetsYZAttr = DenseI64ArrayAttr.get(offsets_y_z)


            sizesXAttr = DenseI64ArrayAttr.get(sizes_x)
            sizesYAttr = DenseI64ArrayAttr.get(sizes_y)
            sizesZAttr = DenseI64ArrayAttr.get(sizes_z)

            result_type_x = RankedTensorType.get(sizes_x, dataType)
            result_type_y = RankedTensorType.get(sizes_y, dataType)
            result_type_z = RankedTensorType.get(sizes_z, dataType)

            
            
            
# def H_edge():
#     for j in range(Ny):
#       for k in range(Nz):
#           Hx[Nx,j,k] += (dt / mu0)*((Ey[Nx,j,k+1]-Ey[Nx,j,k])/Dz - (Ez[Nx,j+1,k]-Ez[Nx,j,k])/Dy);
#     for i in range(Nx):
#       for k in range(Nz):
#           Hy[i,Ny,k] += (dt / mu0)*((Ez[i+1,Ny,k]-Ez[i,Ny,k])/Dx - (Ex[i,Ny,k+1]-Ex[i,Ny,k])/Dz);
#     for i in range(Nx):
#       for j in range(Ny):
#           Hz[i,j,Nz] += (dt / mu0)*((Ex[i,j+1,Nz]-Ex[i,j,Nz])/Dy - (Ey[i+1,j,Nz]-Ey[i,j,Nz])/Dx);



            Ex0_z_e = tensor.ExtractSliceOp(
                result_type_z,
                Ex,
                [],
                [],
                [],
                offsets0ZAttr,
                sizesZAttr,
                stridesAttr,
            )
            
            Ex0_y_e = tensor.ExtractSliceOp(
                result_type_y,
                Ex,
                [],
                [],
                [],
                offsets0YAttr,
                sizesYAttr,
                stridesAttr,
            )

            Ex_y_z_e = tensor.ExtractSliceOp(
                result_type_z,
                Ex,
                [],
                [],
                [],
                offsetsYZAttr,
                sizesZAttr,
                stridesAttr,
            )
            
            Ex_z_y_e = tensor.ExtractSliceOp(
                result_type_y,
                Ex,
                [],
                [],
                [],
                offsetsZYAttr,
                sizesYAttr,
                stridesAttr,
            )
            
            Ey0_z_e = tensor.ExtractSliceOp(
                result_type_z,
                Ey,
                [],
                [],
                [],
                offsets0ZAttr,
                sizesZAttr,
                stridesAttr,
            )
            
            Ey0_x_e = tensor.ExtractSliceOp(
                result_type_x,
                Ey,
                [],
                [],
                [],
                offsets0XAttr,
                sizesXAttr,
                stridesAttr,
            )

            Ey_x_z_e = tensor.ExtractSliceOp(
                result_type_z,
                Ey,
                [],
                [],
                [],
                offsetsXZAttr,
                sizesZAttr,
                stridesAttr,
            )
            
            Ey_z_x_e = tensor.ExtractSliceOp(
                result_type_x,
                Ey,
                [],
                [],
                [],
                offsetsZXAttr,
                sizesXAttr,
                stridesAttr,
            )

            Ez0_x_e = tensor.ExtractSliceOp(
                result_type_x,
                Ez,
                [],
                [],
                [],
                offsets0XAttr,
                sizesXAttr,
                stridesAttr,
            )
            
            Ez0_y_e = tensor.ExtractSliceOp(
                result_type_y,
                Ez,
                [],
                [],
                [],
                offsets0YAttr,
                sizesYAttr,
                stridesAttr,
            )

            Ez_y_x_e = tensor.ExtractSliceOp(
                result_type_x,
                Ez,
                [],
                [],
                [],
                offsetsYXAttr,
                sizesXAttr,
                stridesAttr,
            )
            
            Ez_x_y_e = tensor.ExtractSliceOp(
                result_type_y,
                Ez,
                [],
                [],
                [],
                offsetsXYAttr,
                sizesYAttr,
                stridesAttr,
            )
            
            Hz3 = tensor.ExtractSliceOp(
                result_type_z,
                Hz2,
                [],
                [],
                [],
                offsets0ZAttr,
                sizesZAttr,
                stridesAttr,
            )
            
            Hy3 = tensor.ExtractSliceOp(
                result_type_y,
                Hy2,
                [],
                [],
                [],
                offsets0YAttr,
                sizesYAttr,
                stridesAttr,
            )
            
            Hx3 = tensor.ExtractSliceOp(
                result_type_x,
                Hx2,
                [],
                [],
                [],
                offsets0XAttr,
                sizesXAttr,
                stridesAttr,
            )

            
# def H_edge():
#     for j in range(Ny):
#       for k in range(Nz):
#           Hx[Nx,j,k] += (dt / mu0)*((Ey[Nx,j,k+1]-Ey[Nx,j,k])/Dz - (Ez[Nx,j+1,k]-Ez[Nx,j,k])/Dy);
#     for i in range(Nx):
#       for k in range(Nz):
#           Hy[i,Ny,k] += (dt / mu0)*((Ez[i+1,Ny,k]-Ez[i,Ny,k])/Dx - (Ex[i,Ny,k+1]-Ex[i,Ny,k])/Dz);
#     for i in range(Nx):
#       for j in range(Ny):
#           Hz[i,j,Nz] += (dt / mu0)*((Ex[i,j+1,Nz]-Ex[i,j,Nz])/Dy - (Ey[i+1,j,Nz]-Ey[i,j,Nz])/Dx);

            Hz4 = linalg.curl_step(Ex_y_z_e, Ex0_z_e, Ey_x_z_e, Ey0_z_e, Coef_H,
                                        Dy, Dx, outs=[Hz3])
            Hy4 = linalg.curl_step(Ez_x_y_e, Ez0_y_e, Ex_z_y_e, Ex0_y_e, Coef_H,
                                        Dx, Dz, outs=[Hy3])
            Hx4 = linalg.curl_step(Ey_z_x_e, Ey0_x_e, Ez_y_x_e, Ez0_x_e, Coef_H,
                                        Dz, Dy, outs=[Hx3])

            
            
            
            Hz5 = tensor.InsertSliceOp(
                Hz4,
                Hz2,
                [],
                [],
                [],
                offsets0ZAttr,
                sizesZAttr,
                stridesAttr,
            )
            
            Hy5 = tensor.InsertSliceOp(
                Hy4,
                Hy2,
                [],
                [],
                [],
                offsets0YAttr,
                sizesYAttr,
                stridesAttr,
            )
            
            Hx5 = tensor.InsertSliceOp(
                Hx4,
                Hx2,
                [],
                [],
                [],
                offsets0XAttr,
                sizesXAttr,
                stridesAttr,
            )
    
#   for i in range(1,Nx):
#         for j in range(1,Ny):
#             for k in range(1,Nz):
#                 Ex[i,j,k] += (dt / eps0) * ((Hz[i,j,k]-Hz[i,j-1,k])/Dy-(Hy[i,j,k]-Hy[i,j,k-1])/Dz);
#                 Ey[i,j,k] += (dt / eps0) * ((Hx[i,j,k]-Hx[i,j,k-1])/Dz-(Hz[i,j,k]-Hz[i-1,j,k])/Dx);
#                 Ez[i,j,k] += (dt / eps0) * ((Hy[i,j,k]-Hy[i-1,j,k])/Dx-(Hx[i,j,k]-Hx[i,j-1,k])/Dy);

            offsets_1 = [1, 1, 1]
            offsets_1_x = [0, 1, 1]
            offsets_1_y = [1, 0, 1]
            offsets_1_z = [1, 1, 0]


            offsets1Attr = DenseI64ArrayAttr.get(offsets_1)
            offsets1XAttr = DenseI64ArrayAttr.get(offsets_1_x)
            offsets1YAttr = DenseI64ArrayAttr.get(offsets_1_y)
            offsets1ZAttr = DenseI64ArrayAttr.get(offsets_1_z)
            
            Hx0_5 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Hx5,
                [],
                [],
                [],
                offsets1Attr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )

            Hx_z_5 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Hx5,
                [],
                [],
                [],
                offsets1ZAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )
            
            Hx_y_5 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Hx5,
                [],
                [],
                [],
                offsets1YAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )
                            
            Hy0_5 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Hy5,
                [],
                [],
                [],
                offsets1Attr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )

            Hy_z_5 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Hy5,
                [],
                [],
                [],
                offsets1ZAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )
            
            Hy_x_5 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Hy5,
                [],
                [],
                [],
                offsets1XAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )
            
            Hz0_5 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Hz5,
                [],
                [],
                [],
                offsets1Attr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )

            Hz_x_5 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Hz5,
                [],
                [],
                [],
                offsets1XAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )

            Hz_y_5 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Hz5,
                [],
                [],
                [],
                offsets1YAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )

            Ez1 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Ez,
                [],
                [],
                [],
                offsets1Attr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )
            
            Ey1 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Ey,
                [],
                [],
                [],
                offsets1Attr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )
            
            Ex1 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, Nz], f32),
                Ex,
                [],
                [],
                [],
                offsets1Attr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )


#   for i in range(1,Nx):
#         for j in range(1,Ny):
#             for k in range(1,Nz):
#                 Ex[i,j,k] += (dt / eps0) * ((Hz[i,j,k]-Hz[i,j-1,k])/Dy-(Hy[i,j,k]-Hy[i,j,k-1])/Dz);
#                 Ey[i,j,k] += (dt / eps0) * ((Hx[i,j,k]-Hx[i,j,k-1])/Dz-(Hz[i,j,k]-Hz[i-1,j,k])/Dx);
#                 Ez[i,j,k] += (dt / eps0) * ((Hy[i,j,k]-Hy[i-1,j,k])/Dx-(Hx[i,j,k]-Hx[i,j-1,k])/Dy);
        
            Ez2 = linalg.curl_step(Hy0_5, Hy_x_5, Hx0_5, Hx_y_5, Coef_E,
                                        Dx, Dy, outs=[Ez1])
            Ey2 = linalg.curl_step(Hx0_5, Hx_z_5, Hz0_5, Hz_x_5, Coef_E,
                                        Dz, Dx, outs=[Ey1])
            Ex2 = linalg.curl_step(Hz0_5, Hz_y_5, Hy0_5, Hy_z_5, Coef_E,
                                        Dy, Dz, outs=[Ex1])
                 
      
            
            
            
            Ez3 = tensor.InsertSliceOp(
                Ez2,
                Ez,
                [],
                [],
                [],
                offsets1Attr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )
            
            Ey3 = tensor.InsertSliceOp(
                Ey2,
                Ey,
                [],
                [],
                [],
                offsets1Attr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )
            # casted_tensor1 = tensor.CastOp(tensor_type_target, Ey3).result
            # func.CallOp([], "debug_print", [casted_tensor1]) 
            
            Ex3 = tensor.InsertSliceOp(
                Ex2,
                Ex,
                [],
                [],
                [],
                offsets1Attr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, Nz]),
                stridesAttr,
            )
            
            
# def E_edge():
#     for j in range(1,Ny):
#         for k in range(1,Nz):
#             Ex[0,j,k] += (dt / eps0) * ((Hz[0,j,k]-Hz[0,j-1,k])/Dy-(Hy[0,j,k]-Hy[0,j,k-1])/Dz);
#     for i in range(1,Nx):
#         for k in range(1,Nz):
#             Ey[i,0,k] += (dt / eps0) * ((Hx[i,0,k]-Hx[i,0,k-1])/Dz-(Hz[i,0,k]-Hz[i-1,0,k])/Dx);
#     for i in range(1,Nx):
#         for j in range(1,Ny):
#             Ez[i,j,0] += (dt / eps0) * ((Hy[i,j,0]-Hy[i-1,j,0])/Dx-(Hx[i,j,0]-Hx[i,j-1,0])/Dy);
                            
            
            Hx0_z_e = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, 1], f32),
                Hx5,
                [],
                [],
                [],
                offsets1ZAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, 1]),
                stridesAttr,
            )
            
            Hx0_y_e = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, 1, Nz], f32),
                Hx5,
                [],
                [],
                [],
                offsets1YAttr,
                DenseI64ArrayAttr.get([Nx - 1, 1, Nz]),
                stridesAttr,
            )

            Hx_y_z_e = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, 1], f32),
                Hx5,
                [],
                [],
                [],
                offsetsXAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, 1]),
                stridesAttr,
            )
            
            Hx_z_y_e = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, 1, Nz], f32),
                Hx5,
                [],
                [],
                [],
                offsetsXAttr,
                DenseI64ArrayAttr.get([Nx - 1, 1, Nz]),
                stridesAttr,
            )
            
            Hy0_z_e = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, 1], f32),
                Hy5,
                [],
                [],
                [],
                offsets1ZAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, 1]),
                stridesAttr,
            )
            
            Hy0_x_e = tensor.ExtractSliceOp(
                RankedTensorType.get([1, Ny -1, Nz], f32),
                Hy5,
                [],
                [],
                [],
                offsets1XAttr,
                DenseI64ArrayAttr.get([1, Ny - 1, Nz]),
                stridesAttr,
            )

            Hy_x_z_e = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, 1], f32),
                Hy5,
                [],
                [],
                [],
                offsetsYAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, 1]),
                stridesAttr,
            )
            
            Hy_z_x_e = tensor.ExtractSliceOp(
                RankedTensorType.get([1, Ny -1, Nz], f32),
                Hy5,
                [],
                [],
                [],
                offsetsYAttr,
                DenseI64ArrayAttr.get([1, Ny - 1, Nz]),
                stridesAttr,
            )

            Hz0_x_e = tensor.ExtractSliceOp(
                RankedTensorType.get([1, Ny -1, Nz], f32),
                Hz5,
                [],
                [],
                [],
                offsets1XAttr,
                DenseI64ArrayAttr.get([1, Ny - 1, Nz]),
                stridesAttr,
            )
            
            Hz0_y_e = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, 1, Nz], f32),
                Hz5,
                [],
                [],
                [],
                offsets1YAttr,
                DenseI64ArrayAttr.get([Nx - 1, 1, Nz]),
                stridesAttr,
            )

            Hz_y_x_e = tensor.ExtractSliceOp(
                RankedTensorType.get([1, Ny -1, Nz], f32),
                Hz5,
                [],
                [],
                [],
                offsetsZAttr,
                DenseI64ArrayAttr.get([1, Ny - 1, Nz]),
                stridesAttr,
            )
            
            Hz_x_y_e = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, 1, Nz], f32),
                Hz5,
                [],
                [],
                [],
                offsetsZAttr,
                DenseI64ArrayAttr.get([Nx - 1, 1, Nz]),
                stridesAttr,
            )
            
            Ez4 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, Ny - 1, 1], f32),
                Ez3,
                [],
                [],
                [],
                offsets1ZAttr,
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, 1]),
                stridesAttr,
            )
            
            Ey4 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx - 1, 1, Nz], f32),
                Ey3,
                [],
                [],
                [],
                offsets1YAttr,
                DenseI64ArrayAttr.get([Nx - 1, 1, Nz]),
                stridesAttr,
            )
            
            Ex4 = tensor.ExtractSliceOp(
                RankedTensorType.get([1, Ny -1, Nz], f32),
                Ex3,
                [],
                [],
                [],
                offsets1XAttr,
                DenseI64ArrayAttr.get([1, Ny - 1, Nz]),
                stridesAttr,
            )

            
# def E_edge():
#     for j in range(1,Ny):
#         for k in range(1,Nz):
#             Ex[0,j,k] += (dt / eps0) * ((Hz[0,j,k]-Hz[0,j-1,k])/Dy-(Hy[0,j,k]-Hy[0,j,k-1])/Dz);
#     for i in range(1,Nx):
#         for k in range(1,Nz):
#             Ey[i,0,k] += (dt / eps0) * ((Hx[i,0,k]-Hx[i,0,k-1])/Dz-(Hz[i,0,k]-Hz[i-1,0,k])/Dx);
#     for i in range(1,Nx):
#         for j in range(1,Ny):
#             Ez[i,j,0] += (dt / eps0) * ((Hy[i,j,0]-Hy[i-1,j,0])/Dx-(Hx[i,j,0]-Hx[i,j-1,0])/Dy);
            
            
            Ez5 = linalg.curl_step(Hy0_z_e, Hy_x_z_e, Hx0_z_e, Hx_y_z_e, Coef_E,
                                        Dx, Dy, outs=[Ez4])
            Ey5 = linalg.curl_step(Hx0_y_e, Hx_z_y_e, Hz0_y_e, Hz_x_y_e, Coef_E,
                                        Dz, Dx, outs=[Ey4])
            Ex5 = linalg.curl_step(Hz0_x_e, Hz_y_x_e, Hy0_x_e, Hy_z_x_e, Coef_E,
                                        Dy, Dz, outs=[Ex4])
            # casted_tensor1 = tensor.CastOp(tensor_type_target, Ex5).result
            # func.CallOp([], "debug_print", [casted_tensor1]) 
            
            
            Ez6 = tensor.InsertSliceOp(
                Ez5,
                Ez3,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([1, 1, 0]),
                DenseI64ArrayAttr.get([Nx - 1, Ny - 1, 1]),
                stridesAttr,
            )
            
            
            Ey6 = tensor.InsertSliceOp(
                Ey5,
                Ey3,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([1, 0, 1]),
                DenseI64ArrayAttr.get([Nx - 1, 1, Nz]),
                stridesAttr,
            )
            
            # Ex5 = tensor.ExtractSliceOp(
            #     RankedTensorType.get([1, Ny - 1, Nz - 1], f32),
            #     Ex5_0,
            #     [],
            #     [],
            #     [],
            #     DenseI64ArrayAttr.get([0, 0, 0]),
            #     DenseI64ArrayAttr.get([1, Ny - 1, Nz - 1]),
            #     stridesAttr,
            # )
            
            
            Ex6 = tensor.InsertSliceOp(
                Ex5,
                Ex3,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([0, 1, 1]),
                DenseI64ArrayAttr.get([1, Ny - 1, Nz]),
                stridesAttr,
            )
            
            
          
            Ex_e_y_1 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx+1, 2, Nz+1+1], dataType),
                Ex6,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([0, Ny, 0]),
                DenseI64ArrayAttr.get([Nx+1, 2, Nz+1+1]),
                stridesAttr
            )
            fill_x_e_y_1 = linalg.fill(real_zero, outs= [Ex_e_y_1.result])

            Ex7 = tensor.InsertSliceOp(
                fill_x_e_y_1,
                Ex6,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([0, Ny, 0]),
                DenseI64ArrayAttr.get([Nx+1, 2, Nz+1+1]),
                stridesAttr,
            )
            
            
            
            
                   
          
            Ex_e_z_1 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx+1, Ny+1+1, 2], dataType),
                Ex7,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([0, 0, Nz]),
                DenseI64ArrayAttr.get([Nx+1, Ny+1+1, 2]),
                stridesAttr
            )
            fill_x_e_z_1 = linalg.fill(real_zero, outs= [Ex_e_z_1.result])

            Ex7 = tensor.InsertSliceOp(
                fill_x_e_z_1,
                Ex7,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([0, 0, Nz]),
                DenseI64ArrayAttr.get([Nx+1, Ny+1+1, 2]),
                stridesAttr,
            )
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
          
            Ey_e_x_1 = tensor.ExtractSliceOp(
                RankedTensorType.get([2, Ny+1, Nz+1+1], dataType),
                Ey6,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([Nx, 0, 0]),
                DenseI64ArrayAttr.get([2, Ny+1, Nz+1+1]),
                stridesAttr
            )
            fill_y_e_x_1 = linalg.fill(real_zero, outs= [Ey_e_x_1.result])

            Ey7 = tensor.InsertSliceOp(
                fill_y_e_x_1,
                Ey6,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([Nx, 0, 0]),
                DenseI64ArrayAttr.get([2, Ny+1, Nz+1+1]),
                stridesAttr,
            )
            
            
            
            
    
            Ey_e_z_1 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx+1+1, Ny+1, 2], dataType),
                Ey7,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([0, 0, Nz]),
                DenseI64ArrayAttr.get([Nx+1+1, Ny+1, 2]),
                stridesAttr
            )
            fill_y_e_z_1 = linalg.fill(real_zero, outs= [Ey_e_z_1.result])

            Ey8 = tensor.InsertSliceOp(
                fill_y_e_z_1,
                Ey7,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([0, 0, Nz]),
                DenseI64ArrayAttr.get([Nx+1+1, Ny+1, 2]),
                stridesAttr,
            )
            
            
            
            
            
            
            
                  
            
            
            
            
            
            
          
            Ez_e_x_1 = tensor.ExtractSliceOp(
                RankedTensorType.get([2, Ny+1+1, Nz+1], dataType),
                Ez6,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([Nx, 0, 0]),
                DenseI64ArrayAttr.get([2, Ny+1+1, Nz+1]),
                stridesAttr
            )
            fill_z_e_x_1 = linalg.fill(real_zero, outs= [Ez_e_x_1.result])

            Ez7 = tensor.InsertSliceOp(
                fill_z_e_x_1,
                Ez6,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([Nx, 0, 0]),
                DenseI64ArrayAttr.get([2, Ny+1+1, Nz+1]),
                stridesAttr,
            )
            
            
            
            
        
            Ez_e_y_1 = tensor.ExtractSliceOp(
                RankedTensorType.get([Nx+1+1, 2, Nz+1], dataType),
                Ez7,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([0, Ny, 0]),
                DenseI64ArrayAttr.get([Nx+1+1, 2, Nz+1]),
                stridesAttr
            )
            fill_z_e_y_1 = linalg.fill(real_zero, outs= [Ez_e_y_1.result])

            Ez8 = tensor.InsertSliceOp(
                fill_z_e_y_1,
                Ez7,
                [],
                [],
                [],
                DenseI64ArrayAttr.get([0, Ny, 0]),
                DenseI64ArrayAttr.get([Nx+1+1, 2, Nz+1]),
                stridesAttr,
            )
            
            
            
            
            
            
            
            
            
            
        
            
            
            
            
                # scf.YieldOp(results_=[])

            # return Hx5.result, Hy5.result, Hz5.result, Ex6.result, Ey6.result, Ez6.result
            return
        


    return (module)


# def runner(radix: int, Nx: int):


def transformOpConstruction(module1):
    with InsertionPoint(module1.body):
        @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
        def mod():
            # @named_sequence("__transform_main", [any_op_t()], [])
            # def basic(target: any_op_t()):
              
        # module.operation.attributes["transform.with_named_sequence"] = UnitAttr.get()
            named_sequence = transform.NamedSequenceOp(
            "__transform_main",
            [transform.AnyOpType.get()],
            [transform.AnyOpType.get()],
            arg_attrs = [{"transform.consumed": UnitAttr.get()}])
            with InsertionPoint(named_sequence.body):
                cul_name  = "linalg.curl_step"
                # TILE_SIZE = [16, 16, 16]
                TILE_SIZE = [1, 1, 16]
                TILE_SIZE_X = [1, 1, 16]
                TILE_SIZE_Y = [1, 1, 16]
                TILE_SIZE_Z = [1, 1 ,16]
                expected_types = [any_op_t(), any_op_t(), any_op_t(),
                                any_op_t(), any_op_t(), any_op_t(),
                                any_op_t(), any_op_t(), any_op_t(),
                                any_op_t(), any_op_t(), any_op_t()]
                curl_op = structured.MatchOp.match_op_names(named_sequence.bodyTarget, [cul_name])

                split_results = transform.SplitHandleOp(handle=curl_op, results_=expected_types).results
                # H_Step
                tiledx, loopx = structured.TileUsingForallOp(split_results[0], tile_sizes = TILE_SIZE).results
                tiledy, loopy = structured.TileUsingForallOp(split_results[1], tile_sizes = TILE_SIZE).results
                tiledz, loopz = structured.TileUsingForallOp(split_results[2], tile_sizes = TILE_SIZE).results
                
                fused_loop_result_x_y = loop.loop_fuse_sibling(fused_loop = any_op_t(), target = loopx,
                                                        source = loopy)
                fused_loop_result_x_y_z = loop.loop_fuse_sibling(fused_loop = any_op_t(), target = loopz,
                                                        source = fused_loop_result_x_y)
                # H_Edge
                tiledx1, loopx1 = structured.TileUsingForallOp(split_results[3], tile_sizes = TILE_SIZE_Z).results
                tiledy1, loopy1 = structured.TileUsingForallOp(split_results[4], tile_sizes = TILE_SIZE_Y).results
                tiledz1, loopz1 = structured.TileUsingForallOp(split_results[5], tile_sizes = TILE_SIZE_X).results
            
                    
                # H_Step
                tiledx2, loopx2 = structured.TileUsingForallOp(split_results[6], tile_sizes = TILE_SIZE).results
                tiledy2, loopy2 = structured.TileUsingForallOp(split_results[7], tile_sizes = TILE_SIZE).results
                tiledz2, loopz2 = structured.TileUsingForallOp(split_results[8], tile_sizes = TILE_SIZE).results
                fused_loop_result_x_y2 = loop.loop_fuse_sibling(fused_loop = any_op_t(), target = loopx2,
                                                        source = loopy2)
                fused_loop_result_x_y_z2 = loop.loop_fuse_sibling(fused_loop = any_op_t(), target = loopz2,
                                                        source = fused_loop_result_x_y2)            
            
                # H_Edge
                tiledx3, loopx3 = structured.TileUsingForallOp(split_results[9], tile_sizes = TILE_SIZE_Z).results
                tiledy3, loopy3 = structured.TileUsingForallOp(split_results[10], tile_sizes = TILE_SIZE_Y).results
                tiledz3, loopz3 = structured.TileUsingForallOp(split_results[11], tile_sizes = TILE_SIZE_X).results
                
                
                
                curl_tiled = structured.MatchOp.match_op_names(named_sequence.bodyTarget, [cul_name])
                split_results_tiled = transform.SplitHandleOp(handle=curl_tiled, results_=expected_types).results
                fdtd_func_0 = transform.structured.MatchOp.match_op_names(named_sequence.bodyTarget, ["func.func"])

                    
                apply_cse(fdtd_func_0)
                
                # with InsertionPoint(apply_0.patterns):
                    # ApplyFoldUnitExtentDimsViaReshapesPatternsOp()   
                    
    
            
                # vf = structured.VectorizeChildrenAndApplyPatternsOp(
                #     fdtd_func_0,
                #     disable_multi_reduction_to_contract_patterns=True,
                #     disable_transfer_permutation_map_lowering_patterns=True,
                #     vectorize_nd_extract=True,
                #     vectorize_padding=True,
                # )

            
                # VectorizeOp(split_results_tiled[0])
                # VectorizeOp(split_results_tiled[1])
                # VectorizeOp(split_results_tiled[2])
                # # VectorizeOp(split_results_tiled[3])
                # # VectorizeOp(split_results_tiled[4])
                # # VectorizeOp(split_results_tiled[5])
                # VectorizeOp(split_results_tiled[6])
                # VectorizeOp(split_results_tiled[7])
                # VectorizeOp(split_results_tiled[8])
                # # VectorizeOp(split_results_tiled[9])
                # # VectorizeOp(split_results_tiled[10])
                # # VectorizeOp(split_results_tiled[11])
                


                bufferized_func = bufferization.OneShotBufferizeOp(named_sequence.bodyTarget, bufferize_function_boundaries=True,
                                                                    allow_return_allocs_from_loops=True, 
                                                                    allow_unknown_ops=True,
                                                                    function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
                                                                    )
                fdtd_func_1 = transform.structured.MatchOp.match_op_names(bufferized_func, ["func.func"])
                transformed_result = apply_registered_pass(
                    result=any_op_t(),
                    target=fdtd_func_1,
                    pass_name="buffer-deallocation-pipeline",
                    options=None
                )
                
                fdtd_func_2 = transform.structured.MatchOp.match_op_names(bufferized_func, ["func.func"])
                vf = structured.VectorizeChildrenAndApplyPatternsOp(
                    fdtd_func_2,
                    disable_multi_reduction_to_contract_patterns=True,
                    disable_transfer_permutation_map_lowering_patterns=True,
                    vectorize_nd_extract=True,
                    vectorize_padding=True,
                )
                apply_1 = transform.ApplyPatternsOp(vf)
                with InsertionPoint(apply_1.patterns):
                    # ApplyCanonicalizationPatternsOp()
                    ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp()                    
                apply_cse(vf)
                
                hoist_op = HoistRedundantVectorTransfersOp(any_op_t(), vf.result)
                
                apply_2 = transform.ApplyPatternsOp(hoist_op)
                with InsertionPoint(apply_2.patterns):
                    ApplyTilingCanonicalizationPatternsOp()
                    ApplyFoldUnitExtentDimsViaReshapesPatternsOp()   
                    ApplyCanonicalizationPatternsOp()
                apply_cse(hoist_op)
                apply_3 = transform.ApplyPatternsOp(hoist_op)
                with InsertionPoint(apply_3.patterns):
                    ApplyCastAwayVectorLeadingOneDimPatternsOp()
                    ApplyTransferPermutationPatternsOp()
                    ApplyLowerTransferPatternsOp(max_transfer_rank=1)
                    # ToDo lowering_strategy
                    ApplyLowerTransposePatternsOp()
                    ApplyLowerShapeCastPatternsOp()
                    ApplyRankReducingSubviewPatternsOp()
                    # ToDo max_transfer_rank
                    ApplyTransferToScfPatternsOp()
                    ApplyAllocToAllocaOp()

                    

                transform.bufferization.BufferLoopHoistingOp(hoist_op)
                apply_4 = transform.ApplyPatternsOp(hoist_op)
                with InsertionPoint(apply_4.patterns):
                    ApplyFoldMemrefAliasOpsPatternsOp()
                    ApplyCanonicalizationPatternsOp()

                apply_cse(hoist_op)


                transform.YieldOp([named_sequence.bodyTarget])


def fdtd_compilation_pipeline(Nx, Ny, Nz, module, boiler):
    
    ops = module.operation.regions[0].blocks[0].operations
    # mod = Module.parse("\n".join([str(op) for op in ops]) + boiler)
    # print(module)

    mod = Module.parse("\n".join([str(op) for op in ops]) + boiler)
    # print(mod)
    print_to_file(mod, LLVM_PATH + "/fdtd_init" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + ".mlir")

    pm = PassManager("builtin.module")

    pm.add("transform-interpreter")
    pm.add("test-transform-dialect-erase-schedule")
    # pm.add("one-shot-bufferize{bufferize-function-boundaries=1}")
    # pm.add("buffer-deallocation-pipeline")
    pm.add("math-uplift-to-fma")
    pm.add("canonicalize")
    pm.add("cse")
    pm.add("func.func(llvm-request-c-wrappers)")
    pm.add("finalize-memref-to-llvm")
    pm.add("convert-bufferization-to-memref")
    # pm.add("func.func(convert-linalg-to-loops)")
    # pm.add("func.func(lower-affine)")
    # pm.add("func.func(convert-math-to-llvm)")
    # pm.add("func.func(convert-scf-to-cf)")
    # pm.add("func.func(convert-vector-to-llvm)")
    pm.add("canonicalize")
    pm.add("cse")
    pm.add("test-lower-to-llvm")
    pm.add("reconcile-unrealized-casts")
    pm.run(mod.operation)
    print_to_file(mod, LLVM_PATH + "/fdtd_opt" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + ".mlir")

    print_to_file(mod, FILE_PATH + "fdtd" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + ".mlir")
    call_0 = subprocess.run([LLVM_PATH + "/build/bin/mlir-translate", FILE_PATH + "fdtd" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + ".mlir", "-mlir-to-llvmir", "-o", FILE_PATH + "fdtd" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + ".ll"], stdout=subprocess.PIPE, text=True)
    print("mlir compiled")

    call_1 = subprocess.run([LLVM_PATH + "/build/bin/opt", FILE_PATH + "fdtd" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + ".ll", "-S", "-force-vector-width=16", "-march=native", "-mattr=avx512f", "-O3", "-o", FILE_PATH + "fdtd_opt" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + ".ll"], stdout=subprocess.PIPE, text=True)
    print("llvm opted")

    call_2 = subprocess.run([LLVM_PATH + "/build/bin/llc", FILE_PATH + "fdtd_opt" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + ".ll", "-mcpu=icelake-server", "-O3", "-mattr=+avx512f", "-filetype=obj", "-o", FILE_PATH + "fdtd" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + "_cg" + ".o"], stdout=subprocess.PIPE, text=True)
    print("llc opted")

    call_3 = subprocess.run([LLVM_PATH + "/build/bin/llc", FILE_PATH + "fdtd_opt" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + ".ll", "-filetype=asm", "-mcpu=icelake-server", "-x86-asm-syntax=intel", "-O3", "-mattr=+avx512f", "-o", FILE_PATH + "fdtd" + "_" +  str(int(Nx)) + "_" +  str(int(Ny)) +"_" +  str(int(Nz)) + "_cg" + ".s"], stdout=subprocess.PIPE, text=True)
    print("llc asmed")

    #call_3 = subprocess.run(["ar", "rcs", file_path +"lib.a", file_path + "fft.o", file_path + "mk.o"], stdout=subprocess.PIPE, text=True)

    #/mimer/NOBACKUP/groups/snic2022-22-567/isc/1204/llvm-project//build/bin/opt -S  -force-vector-width=8 2.ll -mattr=avx512f -O3 -enable-interleaved-mem-accesses >3.ll

   
    return mod

def runner(Nx: int, Ny: int, Nz: int):
    
    with Context() as ctx, Location.unknown():
    
        shared_libs = [LLVM_PATH + "/build/lib/libmlir_c_runner_utils.so",
                       LLVM_PATH + "/build/lib/libmlir_runner_utils.so"]
        dataType_Numpy = np.float32
        # Initialize E with random values, H with zero
        Ex = np.random.randn(Nx + 1, Ny+1 + 1, Nz+1 + 1).astype(dataType_Numpy); Hx = np.zeros((Nx+1, Ny, Nz)).astype(dataType_Numpy)
        Ey = np.random.randn(Nx+1 + 1, Ny + 1, Nz+1 + 1).astype(dataType_Numpy); Hy = np.zeros((Nx, Ny+1, Nz)).astype(dataType_Numpy)
        Ez = np.random.randn(Nx+1 + 1, Ny+1 + 1, Nz + 1).astype(dataType_Numpy); Hz = np.zeros((Nx, Ny, Nz+1)).astype(dataType_Numpy)


        # Ex = np.ones((Nx + 1, Ny+1 + 1, Nz+1 + 1)).astype(dataType_Numpy); Hx = np.zeros((Nx+1, Ny, Nz+1)).astype(dataType_Numpy)
        # Ey = np.ones((Nx+1 + 1, Ny + 1, Nz+1 + 1)).astype(dataType_Numpy); Hy = np.zeros((Nx, Ny+1, Nz+1)).astype(dataType_Numpy)
        # Ez = np.ones((Nx+1 + 1, Ny+1 + 1, Nz + 1)).astype(dataType_Numpy); Hz = np.zeros((Nx, Ny, Nz+1+1)).astype(dataType_Numpy)




        Ex[: ,0 ,:] , Ex[: , -2 ,:] , Ex[: ,: ,0] , Ex[: ,: , -2] = 0 , 0 , 0 , 0
        Ey[0 ,: ,:] , Ey[ -2 ,: ,:] , Ey[: ,: ,0] , Ey[: ,: , -2] = 0 , 0 , 0 , 0
        Ez[0 ,: ,:] , Ez[ -2 ,: ,:] , Ez[: ,0 ,:] , Ez[: , -2 ,:] = 0 , 0 , 0 , 0
        
        # Ex[: ,0 ,:] , Ex[: , -1 ,:] , Ex[: ,: ,0] , Ex[: ,: , -1] = 0 , 0 , 0 , 0
        # Ey[0 ,: ,:] , Ey[ -1 ,: ,:] , Ey[: ,: ,0] , Ey[: ,: , -1] = 0 , 0 , 0 , 0
        # Ez[0 ,: ,:] , Ez[ -1 ,: ,:] , Ez[: ,0 ,:] , Ez[: , -1, :] = 0 , 0 , 0 , 0
        
        # Ex[: ,: ,:] , Ex[: , : ,:] , Ex[: ,: ,:] , Ex[: ,: , :] = 0 , 0 , 0 , 0
        # Ey[: ,: ,:] , Ey[: , : ,:] , Ey[: ,: ,:] , Ey[: ,: , :] = 0 , 0 , 0 , 0
        # Ez[: ,: ,:] , Ez[: , : ,:] , Ez[: ,: ,:] , Ez[: ,: , :] = 0 , 0 , 0 , 0

        # Ex_result = np.random.randn(Nx, Ny+1, Nz+1).astype(np.float32); Hx_result = np.ones((Nx+1, Ny, Nz)).astype(np.float32)
        # Ey_result = np.random.randn(Nx+1, Ny, Nz+1).astype(np.float32); Hy_result = np.ones((Nx, Ny+1, Nz)).astype(np.float32)
        # Ez_result = np.random.randn(Nx+1, Ny+1, Nz).astype(np.float32); Hz_result = np.ones((Nx, Ny, Nz+1)).astype(np.float32)

        # curl_result = np.ones(((Nx + 1) * 6, (Ny + 1) * 6, (Nz + 1) * 6)).astype(np.float32)

        # Set PEC boundary conditions
        Ex[: ,0 ,:] , Ex[: , -1 ,:] , Ex[: ,: ,0] , Ex[: ,: , -1] = 0 , 0 , 0 , 0
        Ey[0 ,: ,:] , Ey[ -1 ,: ,:] , Ey[: ,: ,0] , Ey[: ,: , -1] = 0 , 0 , 0 , 0
        Ez[0 ,: ,:] , Ez[ -1 ,: ,:] , Ez[: ,0 ,:] , Ez[: , -1 ,:] = 0 , 0 , 0 , 0
        mem_Ex = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Ex)))
        mem_Ey = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Ey)))
        mem_Ez = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Ez)))
        mem_Hx = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Hx)))
        mem_Hy = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Hy)))
        mem_Hz = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Hz)))
        # mem_Ex_result = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Ex_result)))
        # mem_Ey_result = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Ey_result)))
        # mem_Ez_result = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Ez_result)))
        # mem_Hx_result = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Hx_result)))
        # mem_Hy_result = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Hy_result)))
        # mem_Hz_result = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(Hz_result)))
        # mem_curl_result = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(curl_result)))

        # print(Hx)

        ir = test_fdtd_program(NX, NY, NZ)

        transformOpConstruction(ir)
        null_boiler = ""
        
        execution_engine = ExecutionEngine(fdtd_compilation_pipeline(Nx, Ny, Nz, ir, null_boiler), shared_libs=shared_libs)
        print("compiled")
        func_name = "fdtd_" + str(Nx) + "_" + str(Ny) + "_" + str(Nz)
        pos = [(Nx//2,Ny//2,Nz//2),
            (Nx//3,Ny//3,Nz//3),
            (Nx//5,Ny//5,Nz//5)]
        E_list = [[],[],[]]

        start = time.perf_counter()
        for iter in range(0,Nt):
            execution_engine.invoke(func_name,
                                    # mem_curl_result, 
                                    mem_Ex, mem_Ey, mem_Ez,
                                    mem_Hx, mem_Hy, mem_Hz 
                                    )
        # end = time.perf_counter()
        # print(f"Execution time: {end - start} seconds")

            mlir_Ex = rt.ranked_memref_to_numpy(mem_Ex[0])
            mlir_Ey = rt.ranked_memref_to_numpy(mem_Ey[0])
            mlir_Ez = rt.ranked_memref_to_numpy(mem_Ez[0])
            mlir_Hx = rt.ranked_memref_to_numpy(mem_Hx[0])
            mlir_Hy = rt.ranked_memref_to_numpy(mem_Hy[0])
            mlir_Hz = rt.ranked_memref_to_numpy(mem_Hz[0])
            Ex_show = mlir_Ex[0:Nx, 0:Ny+1, 0:Nz+1]
            Ey_show = mlir_Ey[0:Nx+1, 0:Ny, 0:Nz+1]
            Ez_show = mlir_Ez[0:Nx+1, 0:Ny+1, 0:Nz]

            for i, p in enumerate(pos):
                E_list[i].append(Ex_show[p]+Ey_show[p]+Ez_show[p])
        # print(E_list)
            print("Iter: ", iter)
            # print(Ez_show)

        # check_fields(NX, NY, NZ,
        #             Lx, Ly, Lz,
        #             pos, E_list, Nt, DT)



        print("finished")
runner(NX, NY, NZ)