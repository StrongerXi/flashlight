cmake_minimum_required(VERSION 3.10)

set(
  RUNTIME_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/Device.cpp
  )

target_sources(
  flashlight
  PRIVATE
  ${RUNTIME_SOURCES}
  )
