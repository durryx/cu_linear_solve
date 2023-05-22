### setting the enviroment ###
TESTS = symgs_competitor gauss_seidel_cpu gauss_seidel_gpu
DEPFLAGS=-MP -MD
CXXFLAGS=-fsanitize=memory -fsanitize=undefined -fsanitize=address -Wextra -Wall -Werror -Wl,--fatal-warnings -g

### compiling ###
symgs_competitor: symgs_competitor.cu
	nvcc -O3 $^ -o $@
	$(info run ./symgs_competitor ../kmer_V4a.mtx)

gauss_seidel_cpu: gauss_seidel_cpu.cpp
	clang -O3 $^ -o $@
	$(info run ./gauss_seidel_cpu ../kmer_V4a.mtx)

gauss_seidel_cpu: gauss_seidel_gpu.cu
	nvcc -O3 $^ -o $@
	$(info run ./gauss_seidel_gpu ../kmer_V4a.mtx)

all:$(TESTS)

### run ###



### other ###
clean:
	rm -f $(TESTS)

.PHONY: clean