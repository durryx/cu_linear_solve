### setting the enviroment ###
TESTS = gauss_seidel_performance symgs_competitor
DEPFLAGS=-MP -MD
CXXFLAGS=-fsanitize=memory -fsanitize=undefined -fsanitize=address -Wextra -Wall -Werror -Wl,--fatal-warnings -g

### compiling ###
gauss_seidel_performance: gauss_seidel_gpu.cu
	nvcc -g -G -std=c++20 -DDEBUG_MODE=1 $^ -o $@ 
	$(info run ./gauss_seidel_gpu ../kmer_V4a.mtx)

symgs_competitor: symgs_competitor.cu
	nvcc -O3 $^ -o $@
	$(info run ./symgs_competitor ../kmer_V4a.mtx)

all:$(TESTS)

### run ###
# process may be killed check with: sudo dmesg -T| grep -E -i -B100 '57690'


### other ###
clean:
	rm -f $(TESTS)

.PHONY: clean all
