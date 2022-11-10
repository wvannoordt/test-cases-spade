spade   := ${SPADE}
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
flags += -fconcepts-diagnostics-depth=3
ifeq (${sanny},1)
flags += -fsanitize=undefined,address -fstack-protector-all
endif

main:
	${MAKE} -C ${HYWALL} -f makefile
	${cc} --version
	${cc} -std=c++20 -g -O3 ${flags} ${compflags} -I${spade} -I${PTL}/include -I${HYWALL}/include main.cc -o ${target} -L${PTL}/lib -lPTL -L${HYWALL}/lib -lHyWall

run: main
	./${target}

clean:
	${MAKE} -C ${HYWALL} -f makefile clean
	rm -f ${target}
