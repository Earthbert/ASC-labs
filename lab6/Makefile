# Compiler collection setup
NVCC      ?= nvcc
NVCCFLAGS ?=
CPPFLAGS  ?=
LDFLAGS   ?=
LIBS      ?=
EXEC      ?=

# Logged into your fep8.grid.pub.ro user, employ the following

LOCAL = y

ifndef LOCAL

# Partitions
# Note: xl does not support the ncu profiling tool
BUILD_PARTITION    ?= xl
RUN_PARTITION      ?= ucsx
DOWNLOAD_PARTITION ?= nehalem

# Times
RUN_TIME      ?= 00:05:00
BUILD_TIME    ?= 00:02:00
DOWNLOAD_TIME ?= 00:20:00

# Commands
BUILD_CMD    ?= make LOCAL=y
RUN_CMD      ?= ./$(EXEC)
DOWNLOAD_CMD ?= apptainer pull $(IMG_PATH) $(IMG_URL)
LOAD_PREFIX  ?= apptainer exec --env='TMPDIR=$(HOME)' --nv $(IMG_PATH)

# Image config
IMG_TAG  ?= 12.3.2
IMG_URL  ?= docker://gitlab.cs.pub.ro:5050/asc/asc-public/cuda-labs:$(IMG_TAG)
IMG_PATH ?= ~/DO_NOT_DELETE_IMGS/cuda-labs_$(IMG_TAG).sif
# Older images: https://gitlab.cs.pub.ro/asc/asc-public/container_registry/10

compile: $(IMG_PATH)
	@sbatch                                      \
		--time $(BUILD_TIME)                 \
		--partition $(BUILD_PARTITION)       \
		--wrap="$(LOAD_PREFIX) $(BUILD_CMD)"

run: $(EXEC) $(IMG_PATH)
	@sbatch                                    \
		--gres gpu:1                       \
		--time $(RUN_TIME)                 \
		--partition $(RUN_PARTITION)       \
		--wrap="$(LOAD_PREFIX) $(RUN_CMD)"

$(IMG_PATH):
	@mkdir -p $(dir $(IMG_PATH))
	@sbatch                                   \
		--cpus-per-task=8                 \
		--time $(DOWNLOAD_TIME)           \
		--partition $(DOWNLOAD_PARTITION) \
		--wrap="$(DOWNLOAD_CMD)"
	$(info Please wait! It will take a while.)
	@while [ ! -f $(IMG_PATH) ]; do sleep 10; done

# To run on your local machine define LOCAL=y or LOCAL=whatever
else

# Default rule
build: $(EXEC)

# Build each executable
$(EXEC): $(EXEC).cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

endif

clean:
	rm -f $(EXEC) slurm-*.out slurm-*.err profile.ncu-rep
