### setting the enviroment ###
CCX=clang
BINARY=cuSYMGS
DEPFLAGS=-MP -MD


### gather source code files ###
CODEDIRS=. lib
INCDIRS=. ./include/
CPPFILES=$(foreach D,$(CODEDIRS),$(wildcard $(D)/*.cpp))
CPP_OBJECTS=$(patsubst %.cpp,%.o,$(CPPFILES))
CUDAFILES=$(foreach D,$(CODEDIRS),$(wildcard $(D)/*.cu))
CUDA_OBJECTS=$(patsubst %.cu,%.o,$(CUDAFILES))


### gather dependecies ###
LIBS=-lcudart -lcublas
DEPFILES=$(patsubst %.cpp,%.d,$(CPPFILES))
# specific for every system installation
CUDA_LIBDIRS=-L$(CUDA_PATH)/lib64
CUDA_INCDIRS=-I$(CUDA_PATH)/include


### CUDA compilation ###
CUDAFLAGS=-arch=sm_20
%.o: %.cu
    nvcc -dc $(CUDAFLAGS) $< -o $@


### C++ compilation ###
OBJECTS:=$(CPP_OBJECTS)$(CUDA_OBJECTS)
CXXFLAGS=-fsanitize=memory -fsanitize=undefined -fsanitize=address -Wextra -Wall -Werror -Wl,--fatal-warnings -g $(foreach D,$(INCDIRS),-I$(D)) $(OPT) $(DEPFLAGS)
OPT=-O0
%.o:%.cpp
	$(CCX) $(CXXFLAGS) -c -o $@ $<
all: $(BINARY)
$(BINARY): $(OBJECTS)
	$(CCX) -o $@ $^ $(LIBDIRS) $(CUDA_LIBDIRS) $(INCDIRS) $(CUDA_INCDIRS) $(LIBS) $(CXXFLAGS)
-include $(DEPFILES)


### tests ###
test:
# system specific path for compute-sanitizer
	./opt/cuda/extras/compute-sanitizer/compute-sanitizer ./$(BINARY) kmer_V4a.mtx
	cuda-gdb --args $(BINARY) kmer_V4a.mtx
# run unit tests

### other ###
clean:
	rm -rf $(BINARY) $(OBJECTS) $(DEPFILES)

diff:
	$(info The status of the repository, and the volume of per-file changes:)
	@git status
	@git diff --stat

.PHONY: all clean distribute diff test