cmake_minimum_required(VERSION 3.10)

# ----------------------------- OneDNN -----------------------------

# Try to find oneDNN
find_package(DNNL 2.5.2 CONFIG)
if (NOT DNNL_FOUND AND FL_USE_ONEDNN)
  message(FATAL_ERROR
    "oneDNN required to build oneDNN backend: oneDNN was not found")
else()
  message(STATUS "oneDNN found")
endif()

# ----------------------------- Sources -----------------------------
find_package(MKL REQUIRED)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnCPUStream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnTensor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
)

target_include_directories(
  flashlight
  PUBLIC
  ${MKL_INCLUDE_DIR}
)

target_link_libraries(
  flashlight
  PUBLIC
  DNNL::dnnl
  ${MKL_LIBRARIES}
)
