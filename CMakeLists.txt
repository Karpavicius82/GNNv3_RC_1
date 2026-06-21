cmake_minimum_required(VERSION 3.20)
project(GNNv2 CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/tools)

# every contract in tools/ and every research probe builds as a standalone executable
file(GLOB CONTRACTS CONFIGURE_DEPENDS tools/*.cpp)
file(GLOB PROBES    CONFIGURE_DEPENDS research/*.cpp)
foreach(src ${CONTRACTS} ${PROBES})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
endforeach()
