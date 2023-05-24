### setting the enviroment ###
CCX := clang++
BINARY := linear_solve
DEPFLAGS := -MP -MD
BUILD_DIR := ./build
SRC_DIR := ./src
TESTS_DIR := ./test
CUDA_PATH := ${CUDA_PATH}
THREADS = $(shell nproc --all)

### compilation flags ###
CXXFLAGS := -Wextra -Wall # -flto
debug: CXXFLAGS += -g3 # -fsanitize=undefined -fsanitize=address # -fsanitize=memory  
debug: SHARED_FLAGS := -g -DDEBUG
debug: $(BUILD_DIR)/$(BINARY)

### gather source code files ###
SRC_FILES := $(shell find $(SRC_DIR) -name '*.cpp' -or -name '*.cu')
OBJS := $(SRC_FILES:%=$(BUILD_DIR)/%.o)
INC_DIRS := $(shell find $(SRC_DIR) -type d)
INCLUDE_FLAGS := $(addprefix -I,$(INC_DIRS)) -MMD -MP

### C++ ###
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CCX) $(INCLUDE_FLAGS) $(CXXFLAGS) $(SHARED_FLAGS) -c $< -o $@

### CUDA ###
$(BUILD_DIR)/%.cu.o: %.cu
	mkdir -p $(dir $@)
	nvcc $(INCLUDE_FLAGS) $(SHARED_FLAGS) --dlink-time-opt -dc -dlink $< -o $@ -lcudart -lcublas -lcudadevrt 

# The final build step.
$(BUILD_DIR)/$(BINARY): $(OBJS)
# see https://stackoverflow.com/questions/17278932/cuda-shared-library-linking-undefined-reference-to-cudaregisterlinkedbinary for linking with clang++
	nvcc $(OBJS) $(SHARED_FLAGS) --dlink-time-opt -o $@

-include $(OBJS:.o=.d)


### tests ###
test:
	$(info see ./test directory for tests)
	@echo "execute this: ${CUDA_PATH}/extras/compute-sanitizer/compute-sanitizer ./${BINARY} kmer_V4a.mtx";
	@echo "debug cuda kernerls with: cuda-gdb --args ${BINARY} kmer_V4a.mtx"
	cd test && $(MAKE)

### other ###
clean:
	rm -rf build

diff:
	$(info The status of the repository, and the volume of per-file changes:)
	@git status
	@git diff --stat

.PHONY: clean diff test