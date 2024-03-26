COMPILER=nvcc
LIBS=-lm

all: task5

%: %.cu
	$(COMPILER) $^ -o $@ $(LIBS)

clean:
	rm -rf task5
