#include "includes.h"


int max(int a, int b) {
  if(a>=b)
    return a;
  else
    return b;
}

int main(int argc, char *argv[]) {
  
  int i;

  //initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  //printf("HELLO from MPI process %d of %d\n", my_rank, p);

  get_params(argc, argv);
  //printf("nproc_rows %d nproc_cols %d\n", nproc_rows, nproc_cols);

  //initialize Cblacs
  Cblacs_pinfo(&my_rank, &p);
  Cblacs_get(-1, 0, &blacs_grid);
  Cblacs_gridinit(&blacs_grid, "Row", nproc_rows, nproc_cols);
  Cblacs_gridinfo(blacs_grid, &nproc_rows, &nproc_cols, &my_process_row,
		  &my_process_col);
  
  //node0_printf("HELLO!\n");
  double start = MPI_Wtime();
  /*printf("HELLO from blacs processs %d (%d, %d)  of %d\n",my_rank, my_process_row, 
    my_process_col, p);*/
    
  //printf("BEFORE SETUP on process %d\n", my_rank);
  setup(argc, argv);
  //node0_printf("FINISHED SETUP\n");
  for(i=0; i<warms; i++) {
    update();
  }
  node0_printf("WARMUP FINISHED!\n", my_rank);
  for(i=1; i <= trajecs; i++) {
    update();
    if(i%meas==0)
      measure();
  }
  node0_printf("TRAJECTORIES FINISHED!\n");
  node0_printf("Time elapsed: %e seconds\n", ((double)MPI_Wtime() - start));

  if( strncmp("FORGET", savefile,6) == 0 ) { //save final config
    node0_printf("FINAL CONFIGURATION WILL BE DISCARDED!\n");
  }
  else {
    node0_printf("SAVING MATRICES TO FILE %s\n", savefile);
    if(my_rank==0) //only save matrices from the head node
      save_matrices();
  }

  node0_printf("ACCEPTS = %d\n", accepts);
  node0_printf("REJECTS = %d\n", rejects);

  //free allocated variables
  gsl_rng_free(r);
  gsl_matrix_complex_free(V);
  gsl_matrix_complex_free(B);
  gsl_matrix_complex_free(H_B);
  gsl_matrix_complex_free(B_temp);
  gsl_matrix_complex_free(C);
  gsl_matrix_complex_free(H_C);
  gsl_matrix_complex_free(C_temp);
  gsl_matrix_complex_free(W);
  gsl_matrix_complex_free(H_W);
  gsl_matrix_complex_free(W_temp);
  gsl_matrix_complex_free(W_T);
  gsl_matrix_complex_free(H_W_T);
  gsl_matrix_complex_free(W_T_temp);
  gsl_vector_free(eigen_val_old);
  gsl_vector_free(eigen_val);
  free(V2);
  free(eigen_val2);
  free(eigen_val2_old); 

  //close Cblacs and MPI
  Cblacs_gridexit(0);
  MPI_Finalize();

  return 0;

}//main()

/* Constructs matrix V from temp{B,C,W,W_T} */
void construct_V() {

  int i, j;
  gsl_complex temp, temp2;
  MKL_Complex16 temp3,temp4;

  for(i=0; i<(2*size+nu); i++) {
    for(j=0; j<(2*size+nu); j++) {
      if(i<size && j<(size+nu)) { //upper left block W + a*W_T
	temp = gsl_matrix_complex_get(W_T_temp,i,j);
	temp = gsl_complex_mul_real(temp, a_W);
	temp = gsl_complex_add(gsl_matrix_complex_get(W_temp,i,j),temp);
	V2[i+j*(2*size+nu)].real = temp.dat[0];
	V2[i+j*(2*size+nu)].imag = temp.dat[1];
	gsl_matrix_complex_set(V,i,j,temp);
      }
      if(i < size && j >= (size+nu)) { //upper right block a*C
	temp = gsl_matrix_complex_get(C_temp,i,(j-(size+nu)));
	temp = gsl_complex_mul_real(temp, a);
	V2[i+j*(2*size+nu)].real = temp.dat[0];
	V2[i+j*(2*size+nu)].imag = temp.dat[1];
	gsl_matrix_complex_set(V,i,j,temp);
      }
      if(i >= size && j < (size+nu)) { //lower left block a*B
	temp = gsl_matrix_complex_get(B_temp,(i-size),j);
	temp = gsl_complex_mul_real(temp, a);
	V2[i+j*(2*size+nu)].real = temp.dat[0];
	V2[i+j*(2*size+nu)].imag = temp.dat[1];
	gsl_matrix_complex_set(V,i,j,temp);
      }
      if(i >= size && j >= (size+nu)) { //lower right block W_dagger - a*W_T_dagger
	temp = gsl_matrix_complex_get(W_temp,(j-(size+nu)),(i-size));
	temp = gsl_complex_conjugate(temp);
	temp2 = gsl_matrix_complex_get(W_T_temp,(j-(size+nu)),(i-size));
	temp2 = gsl_complex_conjugate(temp2);
	temp2 = gsl_complex_mul_real(temp2, -a_W);
	temp = gsl_complex_add(temp2,temp);
	V2[i+j*(2*size+nu)].real = temp.dat[0];
	V2[i+j*(2*size+nu)].imag = temp.dat[1];
	gsl_matrix_complex_set(V,i,j,temp);
      } 
      temp = gsl_matrix_complex_get(V,i,j);
      //printf("V %d %d = %e, %e ", i, j , temp.dat[0], temp.dat[1]);
      //printf("V2 %d %d = %e. %e ", i, j, V2[i+(2*size+nu)*j].real, V2[i+(2*size+nu)*j].imag);
    }//for j
    //printf("\n");
  }//for i  
  
  return;

}//construct_V()

/* construct the various matrices with appropriate gaussian distribution */
void setup(int argc, char *argv[]) {


  FILE *ifp;
  int read_status;

  //setup random number generator
  r = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(r,seed);
  node0_printf("seed %d\n", seed);
  
  //set Cblacs variables
  int info, size2 = 2*size + nu;
  mp = numroc_(&size2, &row_block_size, &my_process_row, &i_zero, &nproc_rows);
  nq = numroc_(&size2, &col_block_size, &my_process_col, &i_zero, &nproc_cols);
  Cblacs_barrier(blacs_grid, "a");
  //printf("on process %d mp = %d, nq = %d\n", my_rank, mp, nq);
  lld = max(mp, 1);
  MKL_Complex16 *V_inv_distr = (MKL_Complex16 *)malloc(mp*nq*sizeof(MKL_Complex16));
  MKL_Complex16 *VV_dag_distr = (MKL_Complex16 *)malloc(mp*nq*sizeof(MKL_Complex16));
  MKL_Complex16 *Z_distr = (MKL_Complex16 *)malloc(mp*nq*sizeof(MKL_Complex16));
  descinit_(descZ_distr, &size2, &size2, &row_block_size, &col_block_size, &i_zero,
	    &i_zero, &blacs_grid, &lld, &info);
  descinit_(descV_inv_distr, &size2, &size2, &row_block_size, &col_block_size, &i_zero,
	    &i_zero, &blacs_grid, &lld, &info);
  descinit_(descVV_dag_distr, &size2, &size2, &row_block_size, &col_block_size, &i_zero,
	    &i_zero, &blacs_grid, &lld, &info);

  double *w = (double *)malloc(size2*sizeof(double));
  MKL_Complex16 *work = (MKL_Complex16 *)malloc(1*sizeof(MKL_Complex16));
  MKL_INT lwork = -1;
  MKL_Complex16 *rwork = (MKL_Complex16 *)malloc(1*sizeof(MKL_Complex16));
  MKL_INT lrwork = -1;
  //printf("before pzheev on process %d\n", my_rank);
  /*if(my_rank == 0) {
    printf("size2 = %d\n", size2);
    
    }*/
  pzheev_( "N", "L", &size2, V_inv_distr, &i_one, &i_one, descV_inv_distr, w, Z_distr,
    &i_one, &i_one, descZ_distr, work, &lwork, rwork, &lrwork, &info);
  
  /*printf("on process %d, info after first pzheev %d %e %e\n", my_rank, info, work[0].real,
    rwork[0].real);*/

  lwork_ev = (int)work[0].real;
  lrwork_ev = (int)rwork[0].real;

  free(work);
  free(rwork);
  free(w);

  int nPivotIndices[2*max(mp,nq)];
  lwork = -1;
  int liwork = -1;
  work = (MKL_Complex16 *)malloc(1*sizeof(MKL_Complex16));
  int *iwork = (int *)malloc(1*sizeof(int));

  pzgetri_( &size2, VV_dag_distr, &i_one, &i_one, descVV_dag_distr, &nPivotIndices[0], work, &lwork, 
	   iwork, &liwork, &info);

  lwork_lu = (int)work[0].real;
  liwork_lu = iwork[0];

  free(work);
  free(iwork);

  free(V_inv_distr);
  free(VV_dag_distr);
  free(Z_distr);

  //allocate space for matrices
  V = gsl_matrix_complex_alloc((2*size+nu), (2*size+nu));
  V2 = malloc((2*size+nu)*(2*size+nu)*sizeof(MKL_Complex16));
  //node0_printf("INIT 1\n");
  W = gsl_matrix_complex_alloc(size, (size+nu));
  H_W = gsl_matrix_complex_alloc(size, (size+nu));
  W_temp = gsl_matrix_complex_alloc(size, (size+nu));
  W_T = gsl_matrix_complex_alloc(size, (size+nu));
  H_W_T = gsl_matrix_complex_alloc(size, (size+nu));
  W_T_temp = gsl_matrix_complex_alloc(size, (size+nu));
  B = gsl_matrix_complex_alloc((size+nu), (size+nu));
  H_B = gsl_matrix_complex_alloc((size+nu), (size+nu));
  B_temp = gsl_matrix_complex_alloc((size+nu), (size+nu));
  C = gsl_matrix_complex_alloc(size, size);
  H_C = gsl_matrix_complex_alloc(size, size);
  C_temp = gsl_matrix_complex_alloc(size, size);
  //allocate eigenvalues
  eigen_val = gsl_vector_alloc(2*size+nu);
  eigen_val_old = gsl_vector_alloc(2*size+nu);
  eigen_val2 = malloc((2*size+nu)*sizeof(double));
  eigen_val2_old = malloc((2*size+nu)*sizeof(double));

  //copy_matrices();
  //return;

  //DEBUG F_ACTION size=2 nu=1
  /*gsl_matrix_complex_set(V,0,0,gsl_complex_rect(1.,0));
  gsl_matrix_complex_set(V,1,1,gsl_complex_rect(4.,0));
  gsl_matrix_complex_set(V,0,1,gsl_complex_rect(2.,0));
  gsl_matrix_complex_set(V,1,0,gsl_complex_rect(3.,0));
  f_action();  
  node0_printf("finished ferm\n"); 
  entry = 0.;
  for(i=0; i<2; i++) {
    for(j=0; j<3; j++) {
      node0_printf("entry = %e\n", entry);
      gsl_matrix_complex_set(W,i,j,gsl_complex_rect(entry,1.));
      temp  = gsl_matrix_complex_get(W,i,j);
      node0_printf("check W %d %d = %e, %e", i,j,temp.dat[0],temp.dat[1]);
      gsl_matrix_complex_set(W_T,i,j,gsl_complex_rect(entry,1.));
      temp  = gsl_matrix_complex_get(W_T,i,j);
      node0_printf("check W_T %d %d = %e, %e", i,j,temp.dat[0],temp.dat[1]);
      entry += 1.;     
    }
  }
  entry = 0.;
  for(i=0; i<3; i++) 
    for(j=0; j<3; j++) {
      gsl_matrix_complex_set(B,i,j,gsl_complex_rect(entry,1.));
      entry += 1.;
    }
  entry = 0.;
  for(i=0; i<2; i++) 
    for(j=0; j<2; j++)  {
      gsl_matrix_complex_set(C,i,j,gsl_complex_rect(entry,1.));
      entry += 1.;
    }
  g_action();
  return */
    //END DEBUG
  
  if( strncmp("FRESH", loadfile,5) == 0 ) {  //fresh start
    node0_printf("FRESH START!\n");
    node0_printf("Matrix W\n");
    complex_gauss_rand_matrix(size,(size+nu),sqrt(1./(2*size)),W_temp);
    node0_printf("Matrix W_T\n");
    complex_gauss_rand_matrix(size,(size+nu),sqrt(1./(2*size)),W_T_temp);
    node0_printf("Matrix B\n");
    complex_gauss_rand_matrix((size+nu),(size+nu),sqrt(1./(size)), B_temp);
    node0_printf("Matrix C\n");
    complex_gauss_rand_matrix(size,size,sqrt(1./(size)), C_temp);
  }//if 
  else { //readin matrices {B,C,W,W_T}
    node0_printf("LOADING MATRICES FROM FILE %s\n", loadfile);
    ifp = fopen(loadfile, "r");
    if(ifp == NULL) {
      node0_printf("CANNOT OPEN INPUT FILE!\n");
      exit(1);
    }
    //read B
    read_status = gsl_matrix_complex_fread(ifp, B_temp);
    if(read_status != 0) {
      node0_printf("ERROR READING MATRIX B\n");
      exit(1);
    }
    //read C
    read_status = gsl_matrix_complex_fread(ifp, C_temp);
    if(read_status != 0) {
      node0_printf("ERROR READING MATRIX C\n");
      exit(1);
    }
    //read W
    read_status = gsl_matrix_complex_fread(ifp, W_temp);
    if(read_status != 0) {
      node0_printf("ERROR READING MATRIX W\n");
      exit(1);
    }
    //read W_T
    read_status = gsl_matrix_complex_fread(ifp, W_T_temp);
    if(read_status != 0) {
      node0_printf("ERROR READING MATRIX W_T\n");
      exit(1);
    }
  }//else
  copy_matrices();
  construct_V();
 //print_matrices();
  //action_old = d_action();

}//setup()

void get_params(int argc, char *argv[]) {

  int params_int[8];
  float params_float[4];
  int i;

  //get mpi parameters
  nproc_rows = atoi(argv[1]);
  nproc_cols = atoi(argv[2]);
  nprocs=nproc_rows*nproc_cols;
  row_block_size = col_block_size = atoi(argv[3]);
  /*printf("process %d: nproc_rows = %d, nproc_cols = %d, row/col_block_size = %d\n", my_rank, 
    nproc_rows, nproc_cols, row_block_size);*/

  //simulation parameters
  size = atoi(argv[4]); 
  //printf("on process %d size: %d\n", my_rank, size);
  nu = atoi(argv[5]);
  //printf("on process %d nu: %d\n", my_rank, nu);
  nflavors = atoi(argv[6]);
  //printf("on process %d nflavors: %d\n", my_rank, nflavors);
  mass = atof(argv[7]);
  //printf("on process %d mass: %e\n", my_rank, mass);
  a_W = atof(argv[8]);
  //printf("on process %d a_W: %e\n", my_rank, a_W);
  a = atof(argv[9]);
  //printf("on process %d a: %e\n", my_rank, a);
  seed = atoi(argv[10]);
  //printf("on process %d seed: %d\n", my_rank, seed);
  trajecs = atoi(argv[11]);
  //printf("on process %d trajecs: %d\n", my_rank, trajecs);
  warms = atoi(argv[12]);
  //printf("on process %d warms: %d\n", my_rank, warms);
  meas = atoi(argv[13]);
  //printf("on process %d meas: %d\n", my_rank, meas);
  eps = atof(argv[14]);
  //printf("on process %d eps: %e\n", my_rank, eps);
  steps = atoi(argv[15]);
  //printf("on process %d steps: %d\n", my_rank, steps);
  strcpy(loadfile, argv[16]);
  //printf("on process %d loadfile: %s\n", my_rank, loadfile);
  strcpy(savefile, argv[17]);
  //printf("on process %d savefile: %s\n", my_rank, savefile);
  locr_max = (int)((((2*size+nu)/row_block_size)/nproc_rows)+1)*row_block_size;
#ifdef ONE_MASS
  node0_printf("size = %d, nu = %d, nflavors = %d, a = %e, a_W = %e, mass = %e\n", 
	 size, nu, nflavors, a, a_W, mass);
  node0_printf("steps = %d, eps = %e, traj = %d, traj_between_meas = %d\n", steps,
	 eps, trajecs, meas);
  node0_printf("nproc_rows = %d, nproc_cols = %d, nb = %d\n", nproc_rows, nproc_cols, row_block_size);
#else
  node0_printf("size = %d, nu = %d, nflavors = %d, mass1 = %e, mass2 = %e\n", size, 
	 nu, nflavors, mass1, mass2);
#endif

}//get_params()

/* compute action for metropolis step */
double d_action() {

  double ferm_action, gauge_action, mom_action;
  
  gauge_action = g_action();
  ferm_action = f_action();
  mom_action = h_action();
  node0_printf("\n");
  node0_printf("DG = %e DF = %e DH = %e\n", gauge_action-g_action_old, 
	 ferm_action-f_action_old, mom_action-h_action_old);
  
  g_action_old = gauge_action; //store old gauge momentum and fermion actions
  h_action_old = mom_action;
  if( nflavors != 0)
    f_action_old = ferm_action;

  if(nflavors != 0)
    return ((double)gauge_action + ferm_action + mom_action);
  else
    return ((double)gauge_action + mom_action);

}//d_action()

/* Routine that calculates the "fermion action" i.e. Det(D_st + M); 
   assumes that V has been constructed */
double f_action() {

  int i, j;
  //gsl_matrix_complex *temp_mat;
  //gsl_matrix_complex *eigen_vec;
  double temp = 0; double action = 1.;
  //gsl_complex temp1 = gsl_complex_rect(1.,0);

  //calculate fermion action using LAPACK
  int size2=2*size+nu;
  int info;
  MKL_Complex16 *temp_mat2 = (MKL_Complex16 *)malloc(size2*size2*sizeof(MKL_Complex16));
  MKL_Complex16 *VV_dag_distr = (MKL_Complex16 *)malloc(mp*nq*sizeof(MKL_Complex16));
  MKL_Complex16 *Z_distr = (MKL_Complex16 *)malloc(mp*nq*sizeof(MKL_Complex16));
  MKL_Complex16 wkopt;
  MKL_Complex16 *temp2=(MKL_Complex16 *)malloc(size2*size2*sizeof(MKL_Complex16));
  MKL_Complex16 *work2=(MKL_Complex16 *)malloc(lwork_ev*sizeof(MKL_Complex16));
  MKL_Complex16 *rwork = (MKL_Complex16 *)malloc(lrwork_ev*sizeof(MKL_Complex16));
  //gsl_complex temp3, temp1 = gsl_complex_rect(1.,0);
  MKL_Complex16 alpha, beta;
  alpha.real = 1.0; alpha.imag = 0.0;
  beta.real = 0.0; beta.imag = 0.0;
  
  //construct VV_dag
  zgemm("N", "C", &size2, &size2, &size2, &alpha, V2, &size2, V2, &size2, &beta, temp_mat2, &size2);
  
  //distribute VV_dag 
  pblas_Send2D(blacs_grid, size2, size2, temp_mat2, row_block_size, col_block_size, VV_dag_distr, locr_max);
  /*for(i=0; i<size2; i++) {
    for(j=0; j<size2; j++) {
      node0_printf("element %d, %d of V2 = %e, %e\n", i, j, V2[i+j*size2]);
      //pzelset_(VV_dag_distr, &i, &j, descVV_dag_distr, &temp_mat2[(i-1)+(j-1)*size2]);
    }
    }*/

  //printf("Before pzheev_ on process %d (%d, %d)\n", my_rank, my_process_row, my_process_col);
  //calculate eigenvalues with scalapack
  pzheev_( "N", "L", &size2, VV_dag_distr, &i_one, &i_one, descVV_dag_distr, eigen_val2,
	   Z_distr, &i_one, &i_one, descZ_distr, work2, &lwork_ev, rwork, &lrwork_ev, &info);
  /*printf("After pzheev_ on process %d (%d, %d) info = %d\n", my_rank, my_process_row, 
    my_process_col, info);*/
  for(i=0; i<(2*size+nu); i++) {
    //node0_printf("eigenvalue %d = %e\n", i, eigen_val2[i]);
    action *= eigen_val2[i] + mass*mass;
  }
  
  action = -1.0*(double)log(action);
  node0_printf("f_action = %e, ", action);

  //calculate fermion action using GSL (leave for now)
  /*temp_mat = gsl_matrix_complex_alloc( 2*size+nu, 2*size+nu );
  eigen_vec = gsl_matrix_complex_alloc( 2*size+nu, 2*size+nu );
  gsl_matrix_complex_set_zero(temp_mat);
  
  //construct V*V_dagger 
  gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, temp1, V, V, temp1, temp_mat);
  for(i=0; i<(2*size+nu); i++) {
    for(j=0; j<(2*size+nu); j++) {
       temp1 = gsl_matrix_complex_get(temp_mat,i,j);
       //node0_printf("check %d %d = %e %e\n", i, j, temp1.dat[0], temp1.dat[1]);
    }
    }
  //create the workspace
  gsl_eigen_hermv_workspace *work = gsl_eigen_hermv_alloc(2*size+nu);
  //compute eigenvalues
  gsl_eigen_hermv(temp_mat, eigen_val, eigen_vec, work);
  //sort eigenvalues
  gsl_eigen_hermv_sort(eigen_val, eigen_vec, GSL_EIGEN_SORT_VAL_ASC);
  
  // S_f = -lnDet(D_st + m)
  action = 1.;
  for(i=0; i<(2*size+nu); i++) {
    temp = gsl_vector_get(eigen_val,i);
    //node0_printf("eigenvalue %d = %e\n", i, temp);
    //LS_1 += 1/(temp*temp); //First LS Sum Rule < sum_k (1/lambda_k^2) >
    temp = temp + mass*mass;
    action *= temp;
  }
  
  action = -1.0*(double)log(action);
  //node0_printf("f_action = %e\n", action);
  //node0_printf("LS_1 = %e\n", LS_1);
  */
  free(temp_mat2);
  free(temp2);
  free(work2);
  free(rwork);
  free(Z_distr);
  free(VV_dag_distr);
  //gsl_matrix_complex_free(temp_mat);
  //gsl_matrix_complex_free(eigen_vec);
  //gsl_eigen_hermv_free(work);
    
  return action;
  
}//f_action()

/* Routine that calculates the momentum contribution to the total action
   FOR NOW a=0 i.e. 1/2*Tr(H_W*H_W^{dagger}) */
double h_action() {

  int i;
  gsl_matrix_complex *temp_mat;
  gsl_complex temp1 = gsl_complex_rect(1.,0);
  double temp =0; double action = 0.;

  if(a!=0) {

    //calculate 1/2*Tr(H_B*H_B^{dagger})
    temp_mat = gsl_matrix_complex_alloc(size+nu, size+nu);
    gsl_matrix_complex_set_zero(temp_mat); //reset variables
    temp=0;
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, temp1, H_B, H_B, 
		   temp1, temp_mat);
    
    for(i=0; i<size+nu; i++) {
      temp1 = gsl_matrix_complex_get(temp_mat,i,i);
      temp += temp1.dat[0];
    }
    //node0_printf("B contrib. = %e\n", temp);
    temp *= .5;
    action += temp;    

    gsl_matrix_complex_free(temp_mat);

     //calculate 1/2*Tr(H_C*H_C^{dagger})
    temp_mat = gsl_matrix_complex_alloc(size, size);
    gsl_matrix_complex_set_zero(temp_mat); //reset variables
    temp=0; temp1 = gsl_complex_rect(1.,0);
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, temp1, H_C, H_C, 
		   temp1, temp_mat);
    
    for(i=0; i<size; i++) {
      temp1 = gsl_matrix_complex_get(temp_mat,i,i);
      temp += temp1.dat[0];
    }
    //node0_printf("C contrib. = %e\n", temp);
    temp *= .5;
    action += temp;    

    gsl_matrix_complex_free(temp_mat);


  }//if(a!=0)

  if(a_W!=0) {

    //calculate 1/2*Tr(H_W_T*H_W_T^{dagger})
    temp_mat = gsl_matrix_complex_alloc(size, size);
    gsl_matrix_complex_set_zero(temp_mat); //reset variables
    temp=0.; temp1 = gsl_complex_rect(1.,0);
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, temp1, H_W_T, H_W_T, 
		   temp1, temp_mat);
    
    for(i=0; i<size; i++) {
      temp1 = gsl_matrix_complex_get(temp_mat,i,i);
      temp += temp1.dat[0];
    }
    temp *= .5;
    action += temp;    
    //node0_printf("W_T contrib. = %e\n", temp);

    gsl_matrix_complex_free(temp_mat);

  }//if(a_W!=0)
  
  //calculate 1/2*Tr(H_W*H_W^{dagger})
  temp_mat = gsl_matrix_complex_alloc(size, size);
  gsl_matrix_complex_set_zero(temp_mat); //reset variables
  temp=0.; temp1 = gsl_complex_rect(1.,0);
  gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, temp1, H_W, H_W, 
		 temp1, temp_mat);
  
  for(i=0; i<size; i++) {
    temp1 = gsl_matrix_complex_get(temp_mat,i,i);
    temp += temp1.dat[0];
    //node0_printf("W  %d = %e\n", i, temp1.dat[0]);
  }
  temp *= .5;
  action += temp;
  //node0_printf("W contrib. = %e\n", temp);

  gsl_matrix_complex_free(temp_mat);

  node0_printf("h_action = %e", action);

  return action;

}//h_action()


/* Routine that calculates the "gauge action" i.e. P(D_st) 
   from temp matrices */
double g_action() {

  int i;
  gsl_matrix_complex *temp_mat;
  gsl_complex temp1 = gsl_complex_rect(1.,0);
  double temp =0; double action = 0.;


  if( a != 0 ) {
    //calculate C contribution to action: .5*Tr(C_dagger*C)
    temp_mat = gsl_matrix_complex_alloc(size, size);
    gsl_matrix_complex_set_zero(temp_mat);
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, temp1, C_temp, C_temp, 
		   temp1, temp_mat);
    for(i=0; i<size; i++) {
      temp1 = gsl_matrix_complex_get(temp_mat,i,i);
      temp += temp1.dat[0];
    }
    temp *= .5;
    //node0_printf("C contrib. = %e\n", temp);
    action += temp;
    gsl_matrix_complex_free(temp_mat);
    
    //calculate B contribution to action: .5*ReTr(B_dagger*B)
    temp_mat = gsl_matrix_complex_alloc(size+nu, size+nu);
    gsl_matrix_complex_set_zero(temp_mat); //reset variables
    temp1 = gsl_complex_rect(1.,0);
    temp=0;
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, temp1, B_temp, B_temp, 
		   temp1, temp_mat);
    for(i=0; i<(size+nu); i++) {
      temp1 = gsl_matrix_complex_get(temp_mat,i,i);
      temp += temp1.dat[0];
    }
    temp *= .5;
    //node0_printf("B contrib. = %e\n", temp);
    action += temp;
    gsl_matrix_complex_free(temp_mat);

  }//if(a!=0)
  
  if(a_W!=0) {
    //calculate W_T contribution to action: Tr(W_T_dagger*W_T)
    temp_mat = gsl_matrix_complex_alloc(size, size);
    gsl_matrix_complex_set_zero(temp_mat); //reset variables
    temp1 = gsl_complex_rect(1.,0);
    temp=0;
    //node0_printf("TEST BLAS\n");
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, temp1, W_T_temp, W_T_temp, 
		   temp1, temp_mat);
    //node0_printf("BLAS W TEST\n");
    for(i=0; i<size; i++) {
      temp1 = gsl_matrix_complex_get(temp_mat,i,i);
      temp += temp1.dat[0];
    }
    //node0_printf("W_T contrib. = %e\n", temp);
    action += temp;
    gsl_matrix_complex_free(temp_mat);
    
  }//if(a_W!=0)
  
  //calculate W contribution to action: Tr(W_dagger*W)
  temp_mat = gsl_matrix_complex_alloc(size, size);
  gsl_matrix_complex_set_zero(temp_mat); //reset variables
  temp1 = gsl_complex_rect(1.,0);
  temp=0;
  gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, temp1, W_temp, W_temp, 
		 temp1, temp_mat);
  //node0_printf("BLAS W_T TEST\n");
  for(i=0; i<size; i++) {
    temp1 = gsl_matrix_complex_get(temp_mat,i,i);
    temp += temp1.dat[0];
  }
  //node0_printf("W contrib. = %e\n", temp);
  action += temp;

  //node0_printf("before g_action = %e\n", action);
  action = ((double)size)*action;

  gsl_matrix_complex_free(temp_mat);

  node0_printf("g_action = %e, ", action);

  return action;

}//g_action()

/* Generate gaussian distributed complex random momentum matrices
   according to the distribution exp(-1/2*Tr(H*H^{dagger})) */
void ranmom() {
  
  if(a!=0) {
    complex_gauss_rand_matrix((size+nu), (size+nu), 1., B);
    complex_gauss_rand_matrix(size, size, 1., C);
  }//if(a!=0)

  if(a_W!=0) {
    complex_gauss_rand_matrix(size, (size+nu), 1., H_W_T);
  }//if(a_W!=0)

  //generate for H_W
  //node0_printf("H_W: \n");
  complex_gauss_rand_matrix(size, (size+nu), 1., H_W);

}//ranmom()

/* executes one MD trajectory with a Metropolis step at the end */
void update() {

  int i;  
  //node0_printf("IN UPDATE1!\n");
  //generate momenta to begin trajectory
  ranmom();
  //node0_printf("IN UPDATE2!\n");
  //calculate initial action
  action_old = d_action();
  //node0_printf("IN UPDATE3\n");
  for(i=0; i<steps; i++) {
    
    //leap-frog integration of EOM
    update_matrices(eps/2.);
    update_momenta(eps);
    update_matrices(eps/2.); 

  }

  //Metropolis accept/reject step
  metropolis();

  return;
  
}//update()


/* Update matrices by integrating EOM */
void update_matrices(double epsilon) {

  gsl_matrix_complex *temp_mat;
  gsl_complex eps = gsl_complex_rect(epsilon, 0.);

  if(a!=0) {
    
    temp_mat = gsl_matrix_complex_alloc((size+nu), (size+nu));
    
    gsl_matrix_complex_memcpy(temp_mat,H_B);
    gsl_matrix_complex_scale(temp_mat, eps);
    gsl_matrix_complex_add(B_temp, temp_mat);

    gsl_matrix_complex_free(temp_mat);

    temp_mat = gsl_matrix_complex_alloc(size, size);
    
    gsl_matrix_complex_memcpy(temp_mat,H_C);
    gsl_matrix_complex_scale(temp_mat, eps);
    gsl_matrix_complex_add(C_temp, temp_mat);

    gsl_matrix_complex_free(temp_mat);

  }//if(a!=0)

  if(a_W!=0) {

    temp_mat = gsl_matrix_complex_alloc(size, (size+nu));
    
    gsl_matrix_complex_memcpy(temp_mat,H_W_T);
    gsl_matrix_complex_scale(temp_mat, eps);
    gsl_matrix_complex_add(W_T_temp, temp_mat);

    gsl_matrix_complex_free(temp_mat);

  }//if(a_W!=0)

  temp_mat = gsl_matrix_complex_alloc(size, (size+nu));
  gsl_matrix_complex_memcpy(temp_mat,H_W);
  gsl_matrix_complex_scale(temp_mat, eps);
  gsl_matrix_complex_add(W_temp, temp_mat);

  //construct updated V
  construct_V();

  gsl_matrix_complex_free(temp_mat);

}//update_matrices()

/* Update momenta by integrating EOM */
void update_momenta(double epsilon) {

  int i,j,s,info;
  int nPivotIndices[2*max(mp,nq)];
  int size2=2*size+nu;
  MKL_Complex16 *temp_mat2, *work, *V_inv_distr;
  int *iwork;

  gsl_matrix_complex *temp_mat;
  gsl_complex temp_element;
  gsl_complex eps = gsl_complex_rect(epsilon, 0.);
  V_inv = gsl_matrix_complex_alloc(size2, size2);

  V_inv_distr = (MKL_Complex16 *)malloc(mp*nq*sizeof(MKL_Complex16));
  temp_mat2 = (MKL_Complex16 *)malloc(size2*size2*sizeof(MKL_Complex16)); 
  work = (MKL_Complex16 *)malloc(lwork_lu*sizeof(MKL_Complex16));
  iwork = (int *)malloc(liwork_lu*sizeof(int));
  
  /*printf("inside update_momenta1 on process %d (%d, %d)\n", my_rank, my_process_row,
    my_process_col);*/

  //put V into temp_mat2
  for(i=0; i<(2*size+nu); i++) {
    for(j=0; j<(2*size+nu); j++) {
      temp_element = gsl_matrix_complex_get(V, i, j);
      temp_mat2[j*size2+i].real = temp_element.dat[0];
      temp_mat2[j*size2+i].imag = temp_element.dat[1];
    }
  }
  /*printf("inside update_momenta2 on process %d (%d, %d)\n", my_rank, my_process_row,
    my_process_col);*/

  //distribute temp_mat2 (V) into V_inv_distr
   pblas_Send2D(blacs_grid, size2, size2, temp_mat2, row_block_size, col_block_size, V_inv_distr, locr_max);
  /*for(i=1; i<(size2+1); i++) {
    for(j=1; j<(size2+1); j++) {
      //printf("set element %d, %d in A_distr on process %d\n", i, j, my_rank);
      pzelset_(V_inv_distr, &i, &j, descV_inv_distr, &temp_mat2[(i-1)+(j-1)*size2]);
    }
    }*/

  /*printf("inside update_momenta3 on process %d (%d, %d)\n", my_rank, my_process_row,
    my_process_col);*/

  //invert V with result put into temp_mat2
  pzgetrf_( &size2, &size2, V_inv_distr, &i_one, &i_one, descV_inv_distr, &nPivotIndices[0], 
	   &info);
  /*printf("after pzgetrf info1 %d on process %d (%d, %d)\n", info, my_rank, my_process_row,
    my_process_col);*/
  if(info==0)
    pzgetri_( &size2, V_inv_distr, &i_one, &i_one, descV_inv_distr, &nPivotIndices[0], work, 
	     &lwork_lu, iwork, &liwork_lu, &info);

  //but inverse back into temp_mat2
  pblas_Receive2D(blacs_grid, size2, size2, temp_mat2, row_block_size, col_block_size, V_inv_distr,
		  locr_max);
  /*for(i=1; i<(size2+1); i++) {
    for(j=1; j<(size2+1); j++) {
      pzelget_("A", " ", &temp_mat2[(i-1)+(j-1)*size2], V_inv_distr, &i, &j, descV_inv_distr);
      node0_printf("element %d, %d = %e (%e) in A_distr on process %d\n", i, j, 
	A[(i-1)+(j-1)*size2].real, 1./(A[(i-1)+(j-1)*size2].real), my_rank);
    }
  }*/

  //node0_printf("info2 %d\n", info);
  //node0_printf("IN UPDATE_MOM2\n");

  //put result back into V_inv
  for(i=0; i<(2*size+nu); i++) {
    for(j=0; j<(2*size+nu); j++) {
      //node0_printf("%d %d\n", i, j);
      temp_element.dat[0] = temp_mat2[j*(2*size+nu)+i].real;
      temp_element.dat[1] = temp_mat2[j*(2*size+nu)+i].imag;
      //node0_printf("temp_mat2[%d][%d] %e %en", i, j, temp_mat2[j*(2*size+nu)+i].real, temp_mat2[j*(2*size+nu)+i].imag);
      gsl_matrix_complex_set(V_inv, i, j, temp_element);
      //node0_printf("after set complex\n");
    }
    //node0_printf("\n");
  }

  free(temp_mat2);
  free(work);
  free(iwork);
  free(V_inv_distr);

  //invert V
  /*gsl_permutation *p = gsl_permutation_alloc(2*size+nu);
  V_inv = gsl_matrix_complex_alloc((2*size+nu), (2*size+nu));
  temp_mat = gsl_matrix_complex_alloc((2*size+nu), (2*size+nu));

  gsl_matrix_complex_memcpy(temp_mat,V);
  
  gsl_linalg_complex_LU_decomp(temp_mat, p, &s);
  gsl_linalg_complex_LU_invert(temp_mat, p, V_inv);

  gsl_matrix_complex_free(temp_mat);
  gsl_permutation_free(p);*/
  
  if(a!=0) {
    
    //update H_B
    temp_mat = gsl_matrix_complex_alloc((size+nu), (size+nu));
    B_force(temp_mat);
    gsl_matrix_complex_scale(temp_mat, eps);
    gsl_matrix_complex_add(H_B, temp_mat);

    gsl_matrix_complex_free(temp_mat);

    //update H_C
    temp_mat = gsl_matrix_complex_alloc(size, size);
    C_force(temp_mat);
    gsl_matrix_complex_scale(temp_mat, eps);
    gsl_matrix_complex_add(H_C, temp_mat);

    gsl_matrix_complex_free(temp_mat);
    
  }//if(a!=0)
  
  if(a_W!=0) {
    
    //update H_W_T
    temp_mat = gsl_matrix_complex_alloc(size, (size+nu));
    W_T_force(temp_mat);
    gsl_matrix_complex_scale(temp_mat, eps);
    gsl_matrix_complex_add(H_W_T, temp_mat);
    
    gsl_matrix_complex_free(temp_mat);

  }//if(a_W!=0)

  //update H_W
  temp_mat = gsl_matrix_complex_alloc(size, (size+nu));
  W_force(temp_mat);
  gsl_matrix_complex_scale(temp_mat, eps);
  gsl_matrix_complex_add(H_W, temp_mat);

  gsl_matrix_complex_free(temp_mat);
  gsl_matrix_complex_free(V_inv);

}

/* Calculate force for W */
void W_force(gsl_matrix_complex *dest) {

  int i, j;
  gsl_complex temp1, temp2, temp3;
  double temp_real, temp_imag;

  for(i=0; i<size; i++) {
    for(j=0; j<(size+nu); j++) {
      //Calculate force for Re and Im of H_W_{i,j}
      temp1 = gsl_matrix_complex_get(W_temp, i, j);
      temp1 = gsl_complex_mul_real(temp1, -2.*size);
      temp2 = gsl_matrix_complex_get(V_inv, j, i);
      temp3 = gsl_matrix_complex_get(V_inv, (i+size), (j+size+nu));
      if(nflavors != 0 ) { //dynamic fermions
	temp_real = GSL_REAL(temp1) + 2.*GSL_REAL(temp2) + 2.*GSL_REAL(temp3);
	temp_imag = GSL_IMAG(temp1) - 2.*GSL_IMAG(temp2) + 2.*GSL_IMAG(temp3);
      }
      else { //quenched
	temp_real = GSL_REAL(temp1);
	temp_imag = GSL_IMAG(temp1);
      }
      //Place result into dest (W Force)
      temp1 = gsl_matrix_complex_get(dest, i, j);
      GSL_SET_REAL(&temp1, temp_real);
      GSL_SET_IMAG(&temp1, temp_imag);
      gsl_matrix_complex_set(dest,i,j,temp1);
    }
  }

}//W_force()

/* Calculate force for W_T */
void W_T_force(gsl_matrix_complex *dest) {

  int i, j;
  gsl_complex temp1, temp2, temp3;
  double temp_real, temp_imag;

  for(i=0; i<size; i++) {
    for(j=0; j<(size+nu); j++) {
      //Calculate force for Re and Im of H_W_T{i,j}
      temp1 = gsl_matrix_complex_get(W_T_temp, i, j);
      temp1 = gsl_complex_mul_real(temp1, -2.*size);
      temp2 = gsl_matrix_complex_get(V_inv, j, i);
      temp3 = gsl_matrix_complex_get(V_inv, (i+size), (j+size+nu));
      if(nflavors != 0 ) { //dynamic fermions
	temp_real = GSL_REAL(temp1) + a_W*(2.*GSL_REAL(temp2) - 2.*GSL_REAL(temp3));
	temp_imag = GSL_IMAG(temp1) + a_W*(-2.*GSL_IMAG(temp2) - 2.*GSL_IMAG(temp3));
      }
      else { //quenched
	temp_real = GSL_REAL(temp1);
	temp_imag = GSL_IMAG(temp1);
      }
      //Place result into dest (W_T Force)
      temp1 = gsl_matrix_complex_get(dest, i, j);
      GSL_SET_REAL(&temp1, temp_real);
      GSL_SET_IMAG(&temp1, temp_imag);
      gsl_matrix_complex_set(dest,i,j,temp1);
    }
  }
  
}//W_T_force()


/* Calculate force for B */
void B_force(gsl_matrix_complex *dest) { 

  int i, j;
  gsl_complex temp1, temp2;
  double temp_real, temp_imag;

  for(i=0; i<(size+nu); i++) {
    for(j=0; j<(size+nu); j++) {
      //Calculate force for Re and Im of H_BT{i,j}
      temp1 = gsl_matrix_complex_get(B_temp, i, j);
      temp1 = gsl_complex_mul_real(temp1, -size);
      temp2 = gsl_matrix_complex_get(V_inv, j, (i+size));
      if(nflavors != 0 ) { //dynamic fermions
	temp_real = GSL_REAL(temp1) + a*2.*GSL_REAL(temp2);
	temp_imag = GSL_IMAG(temp1) - a*2.*GSL_IMAG(temp2);
      }
      else { //quenched
	temp_real = GSL_REAL(temp1);
	temp_imag = GSL_IMAG(temp1);
      }
      //Place result into dest (B Force)
      temp1 = gsl_matrix_complex_get(dest, i, j);
      GSL_SET_REAL(&temp1, temp_real);
      GSL_SET_IMAG(&temp1, temp_imag);
      gsl_matrix_complex_set(dest,i,j,temp1);
    }
  }


}//B_force()


/* Calculate force for C */
void C_force(gsl_matrix_complex *dest) { 

  int i, j;
  gsl_complex temp1, temp2;
  double temp_real, temp_imag;

  for(i=0; i<size; i++) {
    for(j=0; j<size; j++) {
      //Calculate force for Re and Im of H_C{i,j}
      temp1 = gsl_matrix_complex_get(C_temp, i, j);
      temp1 = gsl_complex_mul_real(temp1, -size);
      temp2 = gsl_matrix_complex_get(V_inv, (j+size+nu), i);
      if(nflavors != 0 ) { //dynamic fermions
	temp_real = GSL_REAL(temp1) + a*2.*GSL_REAL(temp2);
	temp_imag = GSL_IMAG(temp1) - a*2.*GSL_IMAG(temp2);
      }
      else { //quenched
	temp_real = GSL_REAL(temp1);
	temp_imag = GSL_IMAG(temp1);
      }
      //Place result into dest (C Force)
      temp1 = gsl_matrix_complex_get(dest, i, j);
      GSL_SET_REAL(&temp1, temp_real);
      GSL_SET_IMAG(&temp1, temp_imag);
      gsl_matrix_complex_set(dest,i,j,temp1);
    }
  }


}//C_force()


/* populates an MXN complex matrix with real and imaginary parts of matrix
   elements distributed according to <ReA_ij^2> = <ImA_ij^2> = sigma^2 */
void complex_gauss_rand_matrix(int rows, int columns, double sigma, 
			       gsl_matrix_complex *dest) {
  
  int i, j;
  gsl_complex temp;
  //gsl_complex temp2;
  
  for(i=0; i<rows; i++) {
    for(j=0; j<columns; j++) {
      temp.dat[0] = gsl_ran_gaussian(r, sigma);
      temp.dat[1] = gsl_ran_gaussian(r, sigma);
      //node0_printf("%d %d : %e, %e ", i, j, temp.dat[0], temp.dat[1]);
      gsl_matrix_complex_set(dest,i,j,temp);
      //temp2 = gsl_matrix_complex_get(dest,i,j);
      //node0_printf("%d %d temp: real %e imag %e\n", i, j, temp2.dat[0], temp2.dat[1]);
    }
    //node0_printf("\n");
  }

  return;

}//gauss_rand_matrix()

/* does metropolis step action Det(D_st+m)*exp(-S) */
void metropolis() {

  int i;
  double delta_S, action, rand;

  action = d_action();
  delta_S = action - action_old;
  rand = gsl_rng_uniform(r);
  if( exp((double)-delta_S) > rand) { //accept
    action_old = action;
    //copy new matrices V->V_temp=V+dV (need a routine) 2/15/2012
    copy_matrices();
    for(i=0; i<(2*size+nu); i++) { //copy eigenvalues
      //temp = gsl_vector_get(eigen_val,i);
      //gsl_vector_set(eigen_val_old, i, temp);
      eigen_val2_old[i] = eigen_val2[i]; 
    }
    //copy new to old
    //construct_V();
    node0_printf("ACCEPT: delta_s = %e\n", delta_S);
    accepts++;
  }
  else { //reject
    node0_printf("REJECT: delta_s = %e\n", delta_S);
    rejects++;
  }

  return;

}//metropolis()

/* output eigenvalues and compute observables (LS sum rules etc.) */
void measure() {

  int i, j;
  double temp, temp2, LS_1;
  //gsl_matrix_complex *temp_mat;
  //gsl_matrix_complex *eigen_vec;
  //gsl_complex temp3, temp1 = gsl_complex_rect(1.,0);
  int size2=2*size+nu;
  int info, lwork=(2*size+nu)*(2*size+nu);
  MKL_Complex16 *work2, *temp_mat2; 
  MKL_Complex16 wkopt;
  double *rwork;
  MKL_Complex16 alpha, beta;
  alpha.real = 1.0; alpha.imag = 0.0;
  beta.real = 0.0; beta.imag = 0.0;

  //node0_printf("HI MEASURE1\n");
  if(nflavors == 0) { //in quenched case do not use f_action to calc EV
    
    /*temp_mat = gsl_matrix_complex_alloc( 2*size+nu, 2*size+nu );
    eigen_vec = gsl_matrix_complex_alloc( 2*size+nu, 2*size+nu );
    gsl_matrix_complex_set_zero(temp_mat);
    //construct V*V_dagger 
    gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, temp1, V, V, temp1, temp_mat);
    for(i=0; i<(2*size+nu); i++) {
      //node0_printf("HIIIIIIIIIIII\n");
      for(j=0; j<(2*size+nu); j++) {
	temp3 = gsl_matrix_complex_get(temp_mat,i,j);
	node0_printf("V*V_dagger %d %d = %e, %e \n", i, j, temp3.dat[0], temp3.dat[1]);
      }
      }
    //node0_printf("HI MEASURE3\n");
    //create the workspace
    gsl_eigen_hermv_workspace *work = gsl_eigen_hermv_alloc(2*size+nu);
    //compute eigenvalues
    gsl_eigen_hermv(temp_mat, eigen_val_old, eigen_vec, work);
    //sort eigenvalues
    gsl_eigen_hermv_sort(eigen_val_old, eigen_vec, GSL_EIGEN_SORT_VAL_ASC);*/

    //use LAPACK routines to calculate eigenvalues
    temp_mat2 = malloc((2*size+nu)*(2*size+nu)*sizeof(MKL_Complex16));
    //temp_mat3=malloc((2*size+nu)*(2*size+nu)*sizeof(MKL_Complex16));
    work2=malloc((2*size+nu)*(2*size+nu)*sizeof(MKL_Complex16));
    rwork = malloc((3*(2*size+nu)-2)*sizeof(double));
    MKL_Complex16 alpha, beta;
    alpha.real = 1.0; alpha.imag = 0.0;
    beta.real = 0.0; beta.imag = 0.0;
    //first construct V*V_dagger
    zgemm("N", "C", &size2, &size2, &size2, &alpha, V2, &size2, V2, &size2, &beta, temp_mat2, &size2);
    zheev( "N", "L", &size2, temp_mat2, &size2, eigen_val2, work2, &lwork, rwork, &info);

    /*for(i=0; i<(2*size+nu); i++) {
      for(j=0; j<(2*size+nu); j++) { 
	node0_printf("V*V_dagger2 %d %d = %e, %e \n", i, j, temp_mat2[i+j*size2].real, temp_mat2[i+j*size2].imag);
      }
      }*/

    //free variables
    free(temp_mat2);
    free(work2);
    free(rwork);
    /*gsl_matrix_complex_free(temp_mat);
    gsl_matrix_complex_free(eigen_vec);
    gsl_eigen_hermv_free(work);*/
  }

  temp = LS_1 = 0.;

  for(i=0; i<(2*size+nu); i++) {
    if(nflavors == 0) {
      //temp = gsl_vector_get(eigen_val_old,i);
      temp2 = eigen_val2_old[i];
    }
    else {
      //temp = gsl_vector_get(eigen_val,i);
      temp2 = eigen_val2[i];
    }
    //node0_printf("eigenvalue %d = %e\n", i, temp); //eigenvalues of D_st!!!!!
    node0_printf("eigenvalue %d = %e\n", i, sqrt(temp2)+mass); 
    node0_printf("eigenvalue %d = %e\n", i+1, -sqrt(temp2)+mass);
    //node0_printf("eigenvalue %d = %e\n", i+1, -sqrt(temp2));
    LS_1 += 1/(temp2); //First LS Sum Rule < sum_k (1/lambda_k^2) >
  }
  node0_printf("LS_1 = %e\n", LS_1);

  return;

}//measure()


/* copy updated matrices temp{B,C,W,W_T} to {B,C,W,W_T} */
void copy_matrices() {

  int i, j;
  gsl_complex temp;

  //copy B
  //node0_printf("COPY MATRIX B\n");
  for(i=0; i<(size+nu); i++) {
    for(j=0; j<(size+nu); j++ ) {
      temp = gsl_matrix_complex_get(B_temp,i,j);
      //node0_printf("B_temp %d %d = %e %e\n", i, j, temp.dat[0], temp.dat[1]);
      gsl_matrix_complex_set(B,i,j,temp);
      //temp = gsl_matrix_complex_get(B,i,j);
      //node0_printf("B %d %d = %e %e\n", i, j, temp.dat[0], temp.dat[1]);
    }
  }

  //copy C
  for(i=0; i<size; i++) {
    for(j=0; j<size; j++ ) {
      temp = gsl_matrix_complex_get(C_temp,i,j);
      //node0_printf("C_temp %d %d = %e %e\n", i, j, temp.dat[0], temp.dat[1]);
      gsl_matrix_complex_set(C,i,j,temp);
      //temp = gsl_matrix_complex_get(C,i,j);
      //node0_printf("C %d %d = %e %e\n", i, j, temp.dat[0], temp.dat[1]);

    }
  }
  
  //copy W
  for(i=0; i<size; i++) {
    for(j=0; j<(size+nu); j++ ) {
      temp = gsl_matrix_complex_get(W_temp,i,j);
      //node0_printf("W_temp %d %d = %e %e\n", i, j, temp.dat[0], temp.dat[1]);
      gsl_matrix_complex_set(W,i,j,temp);
      //temp = gsl_matrix_complex_get(W,i,j);
      //node0_printf("W %d %d = %e %e\n", i, j, temp.dat[0], temp.dat[1]);
    }
  }

  //copy W_T
  for(i=0; i<size; i++) {
    for(j=0; j<(size+nu); j++ ) {
      temp = gsl_matrix_complex_get(W_T_temp,i,j);
      //node0_printf("W_T_temp %d %d = %e %e\n", i, j, temp.dat[0], temp.dat[1]);
      gsl_matrix_complex_set(W_T,i,j,temp);
      //temp = gsl_matrix_complex_get(W_T,i,j);
      //node0_printf("W_T %d %d = %e %e\n", i, j, temp.dat[0], temp.dat[1]);
    }
  }

  return;

}//copy_matrices()


/* save matrices {B,C,W,W_T} to savefile */
void save_matrices() {

  int write_status;
  FILE *ofp = fopen(savefile, "w");
  if(ofp == NULL) {
    node0_printf("CANNOT OPEN OUTPUT FILE!\n");
    exit(1);
  }
  //save B
  write_status = gsl_matrix_complex_fwrite(ofp, B);
  if(write_status != 0) {
    node0_printf("ERROR WRITING MATRIX B\n");
    exit(1);
  }
  //write C
  write_status = gsl_matrix_complex_fwrite(ofp, C);
  if(write_status != 0) {
    node0_printf("ERROR WRITING MATRIX C\n");
    exit(1);
  }
    //write W
  write_status = gsl_matrix_complex_fwrite(ofp, W);
  if(write_status != 0) {
    node0_printf("ERROR WRITING MATRIX W\n");
    exit(1);
  }
  //write W_T
  write_status = gsl_matrix_complex_fwrite(ofp, W_T);
  if(write_status != 0) {
    node0_printf("ERROR WRITING MATRIX W_T\n");
    exit(1);
  }

  return;

}//save_matrices()

/* debugging routine */
void print_matrices() {

  int i, j;
  gsl_complex temp;

  for(i=0; i<(size+nu); i++) {
    for(j=0; j<(size+nu); j++ ) {
      temp = gsl_matrix_complex_get(B,i,j);
      node0_printf("B %d %d = %e %e", i, j, temp.dat[0], temp.dat[1]);
    }
    node0_printf("\n");
  }

  for(i=0; i<size; i++) {
    for(j=0; j<size; j++ ) {
      temp = gsl_matrix_complex_get(C,i,j);
      node0_printf("C %d %d = %e %e", i, j, temp.dat[0], temp.dat[1]);
    }
    node0_printf("\n");
  }
  
  for(i=0; i<size; i++) {
    for(j=0; j<(size+nu); j++ ) {
      temp = gsl_matrix_complex_get(W,i,j);
      node0_printf("W %d %d = %e %e", i, j, temp.dat[0], temp.dat[1]);
    }
    node0_printf("\n");
  }

  for(i=0; i<size; i++) {
    for(j=0; j<(size+nu); j++ ) {
      temp = gsl_matrix_complex_get(W_T,i,j);
      node0_printf("W_T %d %d = %e %e", i, j, temp.dat[0], temp.dat[1]);
    }
    node0_printf("\n");
  }

  return;

}//print_matrices()
