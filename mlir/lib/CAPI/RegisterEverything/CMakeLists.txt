# Dialect registration.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(test_libs
  ${cuda_test_libs}
  MLIRTestFuncToLLVM
  MLIRAffineTransformsTestPasses
  MLIRArithTestPasses
  MLIRBufferizationTestPasses
  MLIRControlFlowTestPasses
  MLIRDLTITestPasses
  MLIRFuncTestPasses
  MLIRGPUTestPasses
  MLIRLinalgTestPasses
  MLIRLoopLikeInterfaceTestPasses
  MLIRMathTestPasses
  MLIRMemRefTestPasses
  MLIRNVGPUTestPasses
  MLIRSCFTestPasses
  MLIRShapeTestPasses
  MLIRSPIRVTestPasses
  MLIRTensorTestPasses
  MLIRTestAnalysis
  MLIRTestDynDialect
  MLIRTestIR
  MLIRTestOneToNTypeConversionPass
  MLIRTestPass
  MLIRTestPDLL
  MLIRTestReducer
  MLIRTestRewrite
  MLIRTestTransformDialect
  MLIRTestTransforms
  MLIRTilingInterfaceTestPasses
  MLIRVectorTestPasses
  MLIRTestVectorToSPIRV
  MLIRLLVMTestPasses
  MLIRTosaTestPasses
)

add_mlir_upstream_c_api_library(MLIRCAPIRegisterEverything
  RegisterEverything.cpp

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${translation_libs}
  ${conversion_libs}
  ${extension_libs}
  ${test_libs}

  MLIRBuiltinToLLVMIRTranslation
  MLIRCAPIIR
  MLIRLLVMToLLVMIRTranslation
  MLIRCAPITransforms
)
