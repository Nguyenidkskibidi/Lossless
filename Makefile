# ==============================================================================
# PROJECT PAX: Parallel Arithmetic Extreme Archiver
# High-Throughput Lossless Compression Architecture (C++20)
# ==============================================================================

CXX ?= clang++
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Base C++20 and Warning Flags
CXXFLAGS_BASE = -std=c++20 -Wall -Wextra -Wpedantic -pthread -Iinclude

# Architecture-specific hardware acceleration
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        # Apple Silicon (M1/M2/M3/M4) NEON + Hardware CRC32 + Hardware SHA2
        ARCH_FLAGS = -march=native -mcpu=apple-m1 -DPREFIX_DOUBLING_NEON
    else
        ARCH_FLAGS = -march=native
    endif
else
    ARCH_FLAGS = -march=native
endif

# Release Flags with Link-Time Optimization (LTO)
CXXFLAGS_RELEASE = $(CXXFLAGS_BASE) $(ARCH_FLAGS) -O3 -flto -DNDEBUG
CXXFLAGS_DEBUG   = $(CXXFLAGS_BASE) $(ARCH_FLAGS) -O0 -g -DDEBUG

CXXFLAGS ?= $(CXXFLAGS_RELEASE)

SRC = src/sha256.cpp \
      src/crc32.cpp \
      src/tokenizer.cpp \
      src/bwt.cpp \
      src/mtf.cpp \
      src/rans.cpp \
      src/state_table.cpp \
      src/neural_mixer.cpp \
      src/bit_arithmetic.cpp \
      src/cabac.cpp \
      src/engine.cpp

MAIN     = src/main.cpp
TEST_SRC = tests/test_suite.cpp

TARGET      = pax
TEST_TARGET = pax_test

all: $(TARGET) $(TEST_TARGET)

# Fast optimized release binary
release: CXXFLAGS = $(CXXFLAGS_RELEASE)
release: clean all

# Debug binary with symbols
debug: CXXFLAGS = $(CXXFLAGS_DEBUG)
debug: clean all

$(TARGET): $(SRC) $(MAIN)
	@echo "  [CXX] Building PAX Executable ($(TARGET))..."
	$(CXX) $(CXXFLAGS) $(SRC) $(MAIN) -o $(TARGET)
	@echo "  [OK]  $(TARGET) built successfully."

$(TEST_TARGET): $(SRC) $(TEST_SRC)
	@echo "  [CXX] Building PAX Test Suite ($(TEST_TARGET))..."
	$(CXX) $(CXXFLAGS) $(SRC) $(TEST_SRC) -o $(TEST_TARGET)
	@echo "  [OK]  $(TEST_TARGET) built successfully."

test: $(TEST_TARGET) $(TARGET)
	@echo "================================================================="
	@echo "  Running Algorithmic Test Suite..."
	@echo "================================================================="
	./$(TEST_TARGET)
	@echo "================================================================="
	@echo "  Running Automated Bit-Exact Safety Verification..."
	@echo "================================================================="
	./$(TARGET) --test

clean:
	@echo "  [CLEAN] Removing build targets and temporary archives..."
	rm -f $(TARGET) $(TEST_TARGET) *.pax restored_* test_* slice_*

help:
	@echo "PAX Build System:"
	@echo "  make          - Build release binaries (pax, pax_test)"
	@echo "  make release  - Rebuild with full LTO optimization"
	@echo "  make test     - Compile and run complete safety test suite"
	@echo "  make clean    - Remove build artifacts and temporary files"

.PHONY: all release debug test clean help
