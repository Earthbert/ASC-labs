COMPILER=nvcc
LIBS=-lm

all: task1

%: %.cu
	$(COMPILER) $^ -o $@ $(LIBS)

clean:
	rm -rf task1 
