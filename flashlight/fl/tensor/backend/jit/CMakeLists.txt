cmake_minimum_required(VERSION 3.10)

# ----------------------------- Sources -----------------------------

include(${CMAKE_CURRENT_LIST_DIR}/ir/CMakeLists.txt) # ir
include(${CMAKE_CURRENT_LIST_DIR}/matching/CMakeLists.txt) # matching

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/JitBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/JitTensorBase.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Evaluator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Optimizer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
)
