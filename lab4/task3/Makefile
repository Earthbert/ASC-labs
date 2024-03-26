COMPILER=nvcc
LIBS=-lm

all: task3

%: %.cu
	$(COMPILER) $^ -o $@ $(LIBS) -g

clean:
	rm -rf task3
