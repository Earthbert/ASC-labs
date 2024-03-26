COMPILER=nvcc
LIBS=-lm

all: task4

%: %.cu
	$(COMPILER) $^ -o $@ $(LIBS) -g

clean:
	rm -rf task4
