### setting the enviroment ###
CCX := clang++
BINARY :=linear_solve
DEPFLAGS :=-MP -MD
BUILD_DIR :=./build
SRC_DIR :=./src
TESTS_DIR :=./test
CUDA_PATH := ${CUDA_PATH}

### compilation flags ###
CXXFLAGS := -g # -fsanitize=memory -fsanitize=undefined -fsanitize=address -Wextra -Wall -Werror -Wl,--fatal-warnings

### gather source code files ###
SRC_FILES := $(shell find $(SRC_DIR) -name '*.cpp' -or -name '*.cu')
OBJS := $(SRC_FILES:%=$(BUILD_DIR)/%.o)
INC_DIRS := $(shell find $(SRC_DIR) -type d)
INCLUDE_FLAGS := $(addprefix -I,$(INC_DIRS)) -MMD -MP

### C++ ###
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(INCLUDE_FLAGS) $(CXXFLAGS) -c $< -o $@

### CUDA ###
$(BUILD_DIR)/%.cu.o: %.cu
	mkdir -p $(dir $@)
	nvcc $(INCLUDE_FLAGS) $(CXXFLAGS) -dc -dlink $< -o $@ -lcudart -lcublas -lcudadevrt 

# The final build step.
$(BUILD_DIR)/$(BINARY): $(OBJS)
# see https://stackoverflow.com/questions/17278932/cuda-shared-library-linking-undefined-reference-to-cudaregisterlinkedbinary for linking with clang++
	nvcc $(OBJS) -o $@ $(CXXFLAGS)

-include $(OBJS:.o=.d)


### tests ###
test:
	$(info see ./test directory for tests)
	cd test && $(MAKE)
# system specific path for compute-sanitizer
#./opt/cuda/extras/compute-sanitizer/compute-sanitizer ./$(BINARY) kmer_V4a.mtx
#cuda-gdb --args $(BINARY) kmer_V4a.mtx

### other ###
clean:
	rm -rf build

diff:
	$(info The status of the repository, and the volume of per-file changes:)
	@git status
	@git diff --stat

.PHONY: clean diff test