#include "/opt/intel/mkl/include/mkl.h"
#include "mpi.h"
#include <stdio.h>

//distributes A -> A_local for some process (irow, jcol) on the same node e.g. (0,0)

int pblas_2DDistribute(int ictxt, int m, int n, MKL_Complex16 *A, int mb, int nb, MKL_Complex16 *A_local, 
		       int ldal, int irow, int jcol) {

  int i, j, iloc, jloc, ijump, jjump, istart, jstart;
  int nprocr, nprocc, myprocr, myprocc;
  int locr, locc;
  int izero=0;

  Cblacs_gridinfo(ictxt, &nprocr, &nprocc, &myprocr, &myprocc);

  if(irow >= nprocr) { printf("Cyclic Distribute: irow >= nprocr"); return 0; }
  if(jcol >= nprocc) { printf("Cyclic Distribute: jcol >= nprocc"); return 0; }

  //printf("HI1 Distribute for process %d %d\n", irow, jcol);

  locr = numroc_(&m, &mb, &irow, &izero, &nprocr);
  //printf("HI2 Distribute\n");
  locc = numroc_(&n, &nb, &jcol, &izero, &nprocc);

  /*printf("HI3 Distribute for process %d, %d with locr=%d locc=%d\n", irow, jcol,
    locr, locc);*/

  if(locr == 0 || locc == 0) return 0;

  //how much i,j-index of A jumps when we go past a block;
  ijump = (int) (nprocr-1)*mb;
  jjump = (int) (nprocc-1)*nb;

  //compute starting indices
  istart = irow*mb;
  jstart = jcol*nb;

  j = jstart;
  for(jloc=0; jloc<locc; jloc++) {
    //j++;
    i = istart;
    for(iloc=0; iloc<locr; iloc++) {
      //i++;
      if(i < m && j < n) {
	/*printf("On process %d (%d, %d): i=%d, j=%d ilocal=%d jlocal=%d\n", 
	  my_rank, my_process_row, my_process_col, i, j, iloc, jloc);*/
	A_local[iloc+jloc*locr].real = A[i+j*m].real;
	A_local[iloc+jloc*locr].imag = A[i+j*m].imag;
      }
      else {
	/*printf("else On process %d (%d, %d): i=%d, j=%d ilocal=%d jlocal=%d\n", my_rank, 
	  myprocr, myprocc, i, j, iloc, jloc); */
	A_local[iloc+jloc*locr].real = 0.; 
	A_local[iloc+jloc*locr].imag = 0.;
      }
      i++;
      if(i%mb == 0) i = i + ijump;
      //i++;
    }//for iloc
    j++;
    if(j%mb == 0) j = j + jjump;
    //j++;
  } //for jloc
  
  //printf("End of Distribute for process %d, %d\n", irow, jcol);

  return 1;

}//pblas_2DDistribute()

//take distributed A_local -> A on same node e.g. (0, 0)

int pblas_2DGather(int ictxt, int m, int n, MKL_Complex16 *A, int mb, int nb, 
		   MKL_Complex16 *A_local, int ldal, int irow, int jcol) {

  int i, j, ijump, jjump, locr, locc;
  int istart, jstart, iloc, jloc;
  int nprocr, nprocc, myprocr, myprocc;
  int izero=0;
  
  Cblacs_gridinfo(ictxt, &nprocr, &nprocc, &myprocr, &myprocc);
  //printf("on process %d inside pblas_2DGather\n", my_rank);
  if(irow >= nprocr) { printf("Cyclic distribute: irow >= nprocr\n"); return 0; }
  if(jcol >= nprocc) { printf("Cyclic distribute: jcol >= nprocc\n"); return 0; }
  //printf("on process %d inside pblas_2DGather 2\n", my_rank);

  locr = numroc_(&m, &mb, &irow, &izero, &nprocr);
  locc = numroc_(&n, &nb, &jcol, &izero, &nprocc);
  //printf("on process %d inside pblas_2DGather 3\n", my_rank);
  /*printf("on process %d m=%d n=%d mb=%d nb=%d ldal=%d irow=%d jcol=%d\n", my_rank, m, n, mb, nb, 
    ldal, irow, jcol);*/
  /*printf("on process %d locr=%d locc=%d mylocr=%d mylocc=%d\n", my_rank, locr, locc, 
    numroc_(&m, &mb, &myrow, &izero, &nproc_rows), numroc_(&n, &nb, &my_process_col, &izero, &nproc_cols));*/

  if(locr == 0 || locc == 0) return 0;

  ijump = (nprocr-1)*mb;
  jjump = (nprocc-1)*nb;

  istart = irow*mb;
  jstart = jcol*nb;
  //printf("on process %d inside pblas_2DGather 3\n", my_rank);
  j=jstart;
  for(jloc=0; jloc<locc; jloc++) {
    i=istart;
    for(iloc=0; iloc<locr; iloc++) {
      if(i < m && j < n)
	A[i+j*m] = A_local[iloc+jloc*locr];
      i++;
      if(i%mb == 0) i = i + ijump;
      //i++;
    }//for iloc
    j++;
    if(j%nb == 0) j = j + jjump;
    //j++;
  }//for jloc
  //printf("on process %d inside pblas_2DGather 4\n", my_rank);
  return 1;

}//pblas_2DGather()

int pblas_Send2D(int ictxt, int m, int n, MKL_Complex16 *A,
		 int mb, int nb, MKL_Complex16 *A_local, int ldal) {

  int irow, jcol;
  int locr, locc;
  int nprocr, nprocc, myprocr, myprocc;
  int izero=0;

  Cblacs_gridinfo(ictxt, &nprocr, &nprocc, &myprocr, &myprocc);
  //distribute A on A_local of process (0,0), then send to (irow, jcol)
  if(myprocr == 0 && myprocc == 0) { //compute and send on process (0,0)
    for(irow=0; irow<nprocr; irow++) {
      for(jcol=0; jcol<nprocc; jcol++) {
	if( irow != 0 || jcol != 0 ) {
	  //printf("on process %d before first 2DDistribute\n", my_rank);
	  pblas_2DDistribute(ictxt, m, n, A, mb, nb, A_local, ldal, irow, jcol);
	  //printf("on process %d after first 2DDistribute\n", my_rank);
	  locr = numroc_(&m, &mb, &irow, &izero, &nprocr);
	  locc = numroc_(&n, &nb, &jcol, &izero, &nprocc);
	  if(locr > 0 && locc > 0) {
	    //printf("on process %d before zgesd2d\n", my_rank);
	    zgesd2d_(&ictxt, &locr, &locc, A_local, &locr, &irow, &jcol);
	    //printf("on process %d after zgesd2d\n", my_rank);
	  }
	}

      }//for jcol
    }//for irow
  }//if(my_process_row == 0 && my_process_col == 0)
  else { //if not process (0,0), then recieve
    locr = numroc_(&m, &mb, &myprocr, &izero, &nprocr);
    locc = numroc_(&n, &nb, &myprocc, &izero, &nprocc);
    if(locr > 0 && locc > 0) {
      //Cblacs_barrier(ictxt, 'a');
      //printf("on process %d (%d,%d) before zgerv2d_\n", my_rank, my_process_col, my_process_row);
      zgerv2d_(&ictxt, &locr, &locc, A_local, &locr, &izero, &izero);
      //printf("on process %d (%d,%d) after zgerv2d_\n", my_rank, my_process_col, my_process_row);
    }
  }//else

  //compute A_local to stay on node (0,0)
  //printf("on process %d before first Cblacs_barrier\n", my_rank);
  Cblacs_barrier(ictxt, "a");
  //printf("on process %d after first Cblacs_barrier\n", my_rank);
  if(myprocr == 0 && myprocc == 0) {
    pblas_2DDistribute(ictxt, m, n, A, mb, nb, A_local, ldal, 0, 0);
  }
  //printf("on process %d before second Cblacs_barrier\n", my_rank);
  Cblacs_barrier(ictxt, "a"); //sync everything
  //printf("on process %d after second Cblacs_barrier\n", my_rank);

  return 1;

}//pblas_Send2D()

int pblas_Receive2D(int ictxt, int m, int n, MKL_Complex16 *A, 
		    int mb, int nb, MKL_Complex16 *A_local, int ldal) {

  int irow, jcol;
  int locr, locc;
  int nprocr, nprocc, myprocr, myprocc;
  int izero=0;

  //first put A_local of (0, 0) to A
  //printf("on process %d before first 2DGather\n", my_rank);
  Cblacs_gridinfo(ictxt, &nprocr, &nprocc, &myprocr, &myprocc);
  if(myprocr == 0 && myprocc == 0) 
    pblas_2DGather(ictxt, m, n, A, mb, nb, A_local, ldal, 0, 0);
  //printf("on process %d before first barrier\n", my_rank);
  Cblacs_barrier(ictxt, "a"); //sync everything
  //printf("on process %d after first barrier\n", my_rank);

  //receive A_local on process (0, 0) from (irow, jcol)
  if(myprocr == 0 && myprocc == 0) {
    for(irow=0; irow<nprocr; irow++) {
      for(jcol=0; jcol<nprocc; jcol++) {
	locr = numroc_(&m, &mb, &irow, &izero, &nprocr);
	locc = numroc_(&n, &nb, &jcol, &izero, &nprocc);
	if(irow != 0 || jcol != 0) {
	  if(locr > 0 && locc > 0)
	    zgerv2d_(&ictxt, &locr, &locc, A_local, &locr, &irow, &jcol);
	  pblas_2DGather(ictxt, m, n, A, mb, nb, A_local, ldal, irow, jcol);
	}//if
      }//for jcol
    }//for irow
  }//if my_process_row = my_process_col = 0
  else { //send A_local from other processes to (0, 0)
    locr = numroc_(&m, &mb, &myprocr, &izero, &nprocr);
    locc = numroc_(&n, &nb, &myprocc, &izero, &nprocc);
    if( irow != 0 || jcol != 0) 
      zgesd2d_(&ictxt, &locr, &locc, A_local, &locr, &izero, &izero);
  }

  return 1;

}//pblas_Recieve2D()
