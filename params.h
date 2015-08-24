/* function declarations and global variables for rsrmt */
#define ONE_MASS
#define node0_printf if(my_rank==0) printf

//variables
int size; //size of matrix
int nu; //number of zero modes
int nflavors; //number of staggered flavors
#ifdef ONE_MASS
double mass;
#else
double mass1;
double mass2;
#endif
float a_W;
float a;
int seed;
int trajecs;
int warms;
int meas;
double eps;
int steps;
int load_flag; 
int save_flag;
//make sure savefile is empty and loadfile has only been written to ONCE
char loadfile[50] = "fresh                   "; //CRUDE I/O FOR NOW (NEED TO RECOMPILE EACH TIME)
char savefile[50] = "fresht                  "; //-n-nu-nf-mass-a.config
int accepts=0;
int rejects=0;
gsl_matrix_complex *H_W, *H_W_T, *H_B, *H_C;
gsl_matrix_complex *B, *C, *W, *W_T, *V, *V_inv;
MKL_Complex16 *V2;
gsl_matrix_complex *B_temp, *C_temp, *W_temp, *W_T_temp;
gsl_vector *eigen_val;
gsl_vector *eigen_val_old;
double *eigen_val2;
double *eigen_val2_old;
gsl_rng * r;
double action_old;
double f_action_old=0.;
double g_action_old=0.;
double h_action_old=0.;

//scalapack variables
int nproc_rows, nproc_cols;
int row_block_size, col_block_size, locr_max;
int my_process_row, my_process_col;
int blacs_grid;
int p, my_rank;
int iam, nprocs;
int mp, nq;
int lld;
int lwork_ev, lrwork_ev;
int lwork_lu, liwork_lu;
int descV_inv_distr[9], descVV_dag_distr[9], descZ_distr[9];
const int i_zero = 0, i_one = 1;

//routines 
void setup(int argc, char *argv[]);
void get_params(int argc, char *argv[]);
void update();
void update_matrices(double epsilon);
void update_momenta(double epsilon);
void W_force(gsl_matrix_complex *dest);
void W_T_force(gsl_matrix_complex *dest);
void B_force(gsl_matrix_complex *dest);
void C_force(gsl_matrix_complex *dest);
void ran_mom();
void metropolis();
void measure();
void copy_matrices();
void save_matrices();
void construct_V();
void print_matrices();
double d_action();
double f_action();
double g_action();
double h_action();
void free_variables();
void complex_gauss_rand_matrix(int rows, int columns, double sigma, 
			       gsl_matrix_complex *dest);
int pblas_Send2D(int ictxt, int m, int n, MKL_Complex16 *A,
		 int mb, int nb, MKL_Complex16 *A_local, int ldal);
int pblas_Receive2D(int ictxt, int m, int n, MKL_Complex16 *A, 
		    int mb, int nb, MKL_Complex16 *A_local, int ldal);
int pblas_2DDistribute(int ictxt, int m, int n, MKL_Complex16 *A, int mb, int nb,
		       MKL_Complex16 *A_local, int ldal, int irow, int jcol);
int pblas_2DGather(int ictxt, int m, int n, MKL_Complex16 *A, int mb, int nb, 
		   MKL_Complex16 *A_local, int ldal, int irow, int jcol);
