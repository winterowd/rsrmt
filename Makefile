#Makefile for rooted staggered RMT project

CC =mpicc
OPT =
HEADERS =params.h includes.h
MKLPATH =/opt/intel/composer_xe_2011_sp1.10.319/mkl/lib/intel64
MKLINCLUDE =/opt/intel/composer_xe_2011_sp1.10.319/mkl/include
LIBS = -lgsl -lgslcblas -lm -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_blacs_openmpi_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
LDFLAGS = -L${MKLPATH} ${LIBS}
CFLAGS = -I/usr/local/include/ -I${MKLINCLUDE}

rsrmt_hmc: control_hmc.c pblas2D.c params.h includes.h Makefile
	${CC} ${OPT} -o rsrmt_hmc pblas2D.c control_hmc.c ${LDFLAGS}

test_scalapack: test_scalapack.c Makefile
	${CC} ${OPT} -o test_scalapack test_scalapack.c ${LDFLAGS}

test_pzheev: test_pzheev.c Makefile
	${CC} ${OPT} -o test_pzheev test_pzheev.c ${LDFLAGS}

clean:
	  \rm *.o
