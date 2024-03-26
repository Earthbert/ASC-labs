COMPILER=nvcc
LIBS=-lm

all: task2

%: %.cu
	$(COMPILER) $^ -o $@ $(LIBS)

clean:
	rm -rf task2
