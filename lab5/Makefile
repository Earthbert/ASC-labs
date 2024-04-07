# Compiler collection setup
NVCC      ?= nvcc
NVCCFLAGS ?=
CPPFLAGS  ?=
LDFLAGS   ?=
LIBS      ?=
EXECS     ?=

# Cluster setup
PARTITION      ?= ucsx
RUN_TIME       ?= 00:05:00
BUILD_TIME     ?= 00:05:00
IMG            ?= docker://gitlab.cs.pub.ro:5050/asc/asc-public/cuda-labs:12.3.2
BUILD_CMD      ?= make LOCAL=y
RUN_CMD        ?= ./$${exec}
PROFILE_CMD    ?= ncu --details-all ./$${exec}
CONTAINER_EXEC ?= apptainer exec  --env='TMPDIR=$(HOME)' --nv $(IMG)

# Alternatives
# PARTITION  ?= xl
# Note: xl does not support the ncu profiling tool
#
# IMG        ?= docker://gitlab.cs.pub.ro:5050/asc/asc-public/cuda-labs:1.11.4

# Logged into your fep8.grid.pub.ro user, employ the following two

LOCAL=y

ifndef LOCAL

compile:
	sbatch                      \
		--partition $(PARTITION) \
		--gres gpu:1             \
		--time $(BUILD_TIME)     \
		--wrap="$(CONTAINER_EXEC) $(BUILD_CMD)"

run:
	for exec in $(EXECS);                                  \
	do                                                      \
		test -f $${exec} &&                                 \
		sbatch                                              \
			--partition $(PARTITION)                        \
			--gres gpu:1                                    \
			--time $(RUN_TIME)                              \
			--wrap="$(CONTAINER_EXEC) $(RUN_CMD)";               \
	done

profile:
	for exec in $(EXECS);                                  \
	do                                                      \
		test -f $${exec} &&                                 \
		sbatch                                              \
			--partition $(PARTITION)                        \
			--gres gpu:1                                    \
			--time $(RUN_TIME)                              \
			--wrap="$(CONTAINER_EXEC) $(PROFILE_CMD)";               \
	done

# To run on your local machine define LOCAL=y
else

# Default rule
build: $(EXECS)

# Build each executable
$(EXECS): %: %.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

endif

clean:
	rm -f $(EXECS) slurm-*.out slurm-*.err profile.ncu-rep
