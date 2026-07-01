cmake_minimum_required(VERSION 3.20)
project(GNNv3_RC_1 CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/tools)
enable_testing()

# Every contract in tools/ and every research probe builds as a standalone executable.
# Contract tests are registered with CTest; research probes build but are run manually
# because many are exploratory, long-running, or data-dependent.
file(GLOB CONTRACTS CONFIGURE_DEPENDS tools/*.cpp)
file(GLOB PROBES    CONFIGURE_DEPENDS research/*.cpp)
foreach(src ${CONTRACTS} ${PROBES})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  if(src MATCHES "/tools/" AND name MATCHES "_contract_test$")
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES TIMEOUT 600)
    if(name STREQUAL graph_wave_v3_feeling_gate_contract_test)
      set_tests_properties(${name} PROPERTIES LABELS "gnnv3;feeling")
    endif()
  endif()
endforeach()

# Usage:
#   cmake -S . -B build
#   cmake --build build --config Release
#   ctest --test-dir build -C Release --output-on-failure
#   ctest --test-dir build -C Release -L gnnv3 --output-on-failure
