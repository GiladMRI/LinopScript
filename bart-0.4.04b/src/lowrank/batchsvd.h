
#include <complex.h>

extern void batch_svthresh(long M, long N, long num_blocks, float lambda, complex float dst[num_blocks][N][M]);

extern void batch_svthreshx(long M, long N, long num_blocks, float lambda, complex float dst[num_blocks][N][M], unsigned int option);


