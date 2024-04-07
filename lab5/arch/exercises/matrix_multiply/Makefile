EXECS = $(notdir $(CURDIR))

LDFLAGS = -lstdc++

LOAD_CMD = apptainer exec --nv $(IMG)
IMG      = docker://gitlab.cs.pub.ro:5050/asc/asc-public/cuda-labs:1.9.1

include ../../../Makefile
