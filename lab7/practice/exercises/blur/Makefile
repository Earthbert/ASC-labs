EXEC := $(notdir $(CURDIR))

LDFLAGS := -lm

ifdef LOCAL

all: sequential_blur build

# The default rule for C files puts LDFLAGS too early
# See: https://stackoverflow.com/questions/8492064/undefined-reference-to-exp-on-ubuntu-including-math-h-and-linking-with-lm
sequential_blur: sequential_blur.c
	cc $< -o $@ $(LDFLAGS)

endif

include ../../../Makefile

clean-all: clean
	git clean -Xf
