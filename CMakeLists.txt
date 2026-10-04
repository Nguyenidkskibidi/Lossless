cmake_minimum_required(VERSION 3.20)
project(pax VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -pthread)
endif()

include_directories(include)

set(PAX_SOURCES
    src/sha256.cpp
    src/crc32.cpp
    src/tokenizer.cpp
    src/bwt.cpp
    src/mtf.cpp
    src/rans.cpp
    src/neural_mixer.cpp
    src/bit_arithmetic.cpp
    src/engine.cpp
)

add_library(pax_core STATIC ${PAX_SOURCES})

add_executable(pax src/main.cpp)
target_link_libraries(pax PRIVATE pax_core)

add_executable(pax_test tests/test_suite.cpp)
target_link_libraries(pax_test PRIVATE pax_core)

enable_testing()
add_test(NAME PaxTestSuite COMMAND pax_test)
add_test(NAME PaxCliSafetyCheck COMMAND pax --test)
