cmake_minimum_required(VERSION 3.10)

# ------------------------- Components -------------------------

add_library(
  fl_lib_sequence
  ""
  )

target_include_directories(
  fl_lib_sequence
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  )

# Only build the library if the FL CUDA backend is enabled.
if (NOT (CUDA_FOUND OR FL_USE_CUDA))
  message(
    STATUS
    "CUDA not found or not build - forcing FL_LIBRARIES_USE_CUDA to OFF")
  set(FL_LIBRARIES_USE_CUDA OFF)
elseif(CUDA_FOUND)
  message(
    STATUS
    "CUDA found or not build - forcing FL_LIBRARIES_USE_CUDA to ON")
  set(FL_LIBRARIES_USE_CUDA ON)
endif()

# criterion-library
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

set(FL_LIBRARIES_SEQUENCE_CUDA_SOURCES
  ${FL_LIBRARIES_SEQUENCE_CRITERION_CUDA_SOURCES}
  )
set(FL_LIBRARY_CUDA_SOURCES
  ${FL_LIBRARIES_SEQUENCE_CUDA_SOURCES}
  )

target_include_directories(
  fl_lib_sequence
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  )

if (FL_LIBRARIES_USE_CUDA)
  target_sources(
    fl_lib_sequence
    PRIVATE
    ${FL_LIBRARY_CUDA_SOURCES}
    )

  # CUB
  if(CUDA_VERSION_MAJOR VERSION_LESS 11)
    find_package(cub)
    if (NOT cub_FOUND)
      message(STATUS "Could not find cub - will download from source")
      include(${CMAKE_MODULE_PATH}/BuildCUB.cmake)
      add_dependencies(fl_lib_sequence CUB)
    else ()
      message(STATUS "Found cub: (include: ${cub_INCLUDE_DIRS})")
    endif()
  endif()

  target_include_directories(
    fl_lib_sequence
    PUBLIC
    $<BUILD_INTERFACE:${cub_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )

  target_link_libraries(
    fl_lib_sequence
    PUBLIC
    ${CUDA_LIBRARIES}
    )
endif()

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET fl_lib_sequence)
endif()
