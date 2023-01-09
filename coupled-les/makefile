cvdf   := $(realpath ../../../src)
target := vdf.x

ifndef MPI_ENABLE
MPI_ENABLE := 1
endif

ifeq (${MPI_ENABLE},1)
cc := $(shell which mpicxx)
else
cc := $(shell which g++)
endif

compflags :=
compflags += -DMPI_ENABLE=${MPI_ENABLE}

flags :=
flags += -fconcepts-diagnostics-depth=3 -fmax-errors=4

iflags :=
iflags += -I${PTL}/include
iflags += -I${SPADE}/src

lflags :=
lflags += -L${PTL}/lib -lPTL

main:
	${cc} -std=c++20 -g -O3 ${flags} ${compflags} ${iflags} main.cc -o ${target} ${lflags}

run: main
	./${target}

clean:
	rm -f ${target}
