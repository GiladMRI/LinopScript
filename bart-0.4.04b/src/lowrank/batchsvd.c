/* Copyright 2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 *	2015 Frank Ong <frankong@berkeley.edu>
 *	2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <math.h>

#include "misc/misc.h"

#include "num/flpmath.h"
#include "num/blas.h"
#include "num/lapack.h"
#include "num/linalg.h"

#include "batchsvd.h"

#include "misc/debug.h"



void batch_svthresh(long M, long N, long num_blocks, float lambda, complex float dst[num_blocks][N][M])
{
#pragma omp parallel
    {
	long minMN = MIN(M, N);

	PTR_ALLOC(complex float[minMN][M], U);
	PTR_ALLOC(complex float[N][minMN], VT);
	PTR_ALLOC(float[minMN], S);
	PTR_ALLOC(complex float[minMN][minMN], AA);

#pragma omp for
	for (int b = 0; b < num_blocks; b++) {

		// Compute upper bound | A^T A |_inf

		// FIXME: this is based on gratuitous guess-work about the obscure
		// API of this FORTRAN from ancient times... Is it really worth it?

		blas_csyrk('U', (N <= M) ? 'T' : 'N', (N <= M) ? N : M, (N <= M) ? M : N, 1., M, dst[b], 0., minMN, *AA);

		// lambda_max( A ) <= max_i sum_j | a_i^T a_j |

		float s_upperbound = 0;

		for (int i = 0; i < minMN; i++) {

			float s = 0;

			for (int j = 0; j < minMN; j++)
				s += cabsf((*AA)[MAX(i, j)][MIN(i, j)]);

			s_upperbound = MAX(s_upperbound, s);
		}

		/* avoid doing SVD-based thresholding if we know from
		 * the upper bound that lambda_max <= lambda and the
		 * result must be zero */

		if (s_upperbound < lambda * lambda) {

			mat_zero(N, M, dst[b]);
			continue;
		}

		lapack_svd_econ(M, N, *U, *VT, *S, dst[b]);

		// ggg try: Soft threshold, but not on 1st value
		for (int j = 0; j < N; j++)
				(*VT)[j][0] *= ((*S)[0] < 0) ? 0. : ((*S)[0]);

		// float lambda0 = MAX((*S)[0],0.00000001);
		for (int i = 1; i < minMN; i++) {
			for (int j = 0; j < N; j++) {
				// (*VT)[j][i] = 0;
				(*VT)[j][i] *= ((*S)[i] < lambda) ? 0. : ((*S)[i] - lambda);
				// (*VT)[j][i] *= ((*S)[i] < lambda/lambda0) ? 0. : ((*S)[i] - lambda/lambda0);
			}
		}
		// soft threshold
		// for (int i = 0; i < minMN; i++)
		// 	for (int j = 0; j < N; j++)
		// 		(*VT)[j][i] *= ((*S)[i] < lambda) ? 0. : ((*S)[i] - lambda);
		// end ggg try Soft threshold without first

		blas_matrix_multiply(M, N, minMN, dst[b], *U, *VT);
	}

	PTR_FREE(U);
	PTR_FREE(VT);
	PTR_FREE(S);
	PTR_FREE(AA);
    } // #pragma omp parallel
}

void batch_svthreshx(long M, long N, long num_blocks, float lambda, complex float dst[num_blocks][N][M], unsigned int option)
{
	// debug_printf(DP_DEBUG1,"batch_svthreshx\n");
	PTR_ALLOC(float[num_blocks], SumS);
	bool needCurRegulaizerCostNoLambda= getCurRegulaizerCostNoLambda()<0.0f;
	float CurCostNoLambda=getCurRegulaizerCostNoLambda();
	debug_printf(DP_DEBUG3,"CurCostNoLambda %f\n",CurCostNoLambda);
	CurCostNoLambda=0.0f;

#pragma omp parallel
    {
	long minMN = MIN(M, N);

	PTR_ALLOC(complex float[minMN][M], U);
	PTR_ALLOC(complex float[N][minMN], VT);
	PTR_ALLOC(float[minMN], S);
	PTR_ALLOC(complex float[minMN][minMN], AA);

	
#pragma omp for
	for (int b = 0; b < num_blocks; b++) {

		// Compute upper bound | A^T A |_inf

		// FIXME: this is based on gratuitous guess-work about the obscure
		// API of this FORTRAN from ancient times... Is it really worth it?

		blas_csyrk('U', (N <= M) ? 'T' : 'N', (N <= M) ? N : M, (N <= M) ? M : N, 1., M, dst[b], 0., minMN, *AA);

		// lambda_max( A ) <= max_i sum_j | a_i^T a_j |

		float s_upperbound = 0;

		for (int i = 0; i < minMN; i++) {

			float s = 0;

			for (int j = 0; j < minMN; j++)
				s += cabsf((*AA)[MAX(i, j)][MIN(i, j)]);

			s_upperbound = MAX(s_upperbound, s);
		}

		/* avoid doing SVD-based thresholding if we know from
		 * the upper bound that lambda_max <= lambda and the
		 * result must be zero */

		if (s_upperbound < lambda * lambda) {

			(*SumS)[b]=0;

			mat_zero(N, M, dst[b]);
			continue;
		}

		lapack_svd_econ(M, N, *U, *VT, *S, dst[b]);

		float mag0;

		if(needCurRegulaizerCostNoLambda) {
			float CurSSum=0.0f;
			for (int i = 0; i < minMN; i++) {
				CurSSum += fabs((*S)[i]);
			}
			(*SumS)[b]=CurSSum;
			// debug_printf(DP_DEBUG1,"Calculated [%d] %f\n",b,CurSSum);
		}

		// ggg try: Soft threshold, but not on 1st value
		switch(option) {
			case 0:
				// soft threshold
				for (int i = 0; i < minMN; i++)
					for (int j = 0; j < N; j++)
						(*VT)[j][i] *= ((*S)[i] < lambda) ? 0. : ((*S)[i] - lambda);
				break;
			// ggg try Soft threshold without first
			case 1:
				for (int j = 0; j < N; j++)
					(*VT)[j][0] *= ((*S)[0] < 0) ? 0. : ((*S)[0]);

				for (int i = 1; i < minMN; i++) {
					for (int j = 0; j < N; j++) {
						(*VT)[j][i] *= ((*S)[i] < lambda) ? 0. : ((*S)[i] - lambda);
					}
				}
				break;
			case 2: // completely remove all SVs except the first
				for (int j = 0; j < N; j++)
					(*VT)[j][0] *= ((*S)[0] < 0) ? 0. : ((*S)[0]);

				for (int i = 1; i < minMN; i++) {
					for (int j = 0; j < N; j++) {
						(*VT)[j][i] = 0;
					}
				}
				break;
			case 3: // Soft-thresholding relative to the first 
				for (int j = 0; j < N; j++)
					(*VT)[j][0] *= ((*S)[0] < 0) ? 0. : ((*S)[0]);

				float lambda0 = MAX((*S)[0],0.00000001);
				for (int i = 1; i < minMN; i++) {
					for (int j = 0; j < N; j++) {
						(*VT)[j][i] *= ((*S)[i] < lambda/lambda0) ? 0. : ((*S)[i] - lambda/lambda0);
					}
				}
				break;
			case 4: // completely remove all SVs except the first, and limit V (T2*) to decay 				// (*VT)[0][0] - no change
				mag0=cabsf((*VT)[0][0]);
				(*VT)[0][0] *= ((*S)[0] < 0) ? 0. : ((*S)[0]);
				float mag;
				float pha;
				for (int j = 1; j < N; j++) {
					mag=MIN(cabsf((*VT)[j][0]),mag0);
					// mag=cabsf((*VT)[j][0]);
					pha=cargf((*VT)[j][0]);
					(*VT)[j][0] = mag*cexpf(I*pha);

					(*VT)[j][0] *= ((*S)[0] < 0) ? 0. : ((*S)[0]);
				}

				for (int i = 1; i < minMN; i++) {
					for (int j = 0; j < N; j++) {
						(*VT)[j][i] = 0;
					}
				}
				break;
		}

		blas_matrix_multiply(M, N, minMN, dst[b], *U, *VT);
	}

	// if(getCurRegulaizerCost()<0.0f) {
	// 	float CurCost=lambda*md_z1norm2(N, wdims, MD_STRIDES(N, wdims, CFL_SIZE), tmp);
	// 	// float CurCost=lambda*md_zasum(N, wdims, tmp);
	// 	setCurRegulaizerCost(float);
	// }
	

	PTR_FREE(U);
	PTR_FREE(VT);
	PTR_FREE(S);
	PTR_FREE(AA);
    } // #pragma omp parallel

    if(needCurRegulaizerCostNoLambda) {
		for (int i = 0; i < num_blocks; i++) {
				// float tmp=(*SumS)[i];
				// debug_printf(DP_DEBUG1,"Adding [%d] %f\n",i,tmp);
				CurCostNoLambda += fabs((*SumS)[i]);
			}
		debug_printf(DP_DEBUG1,"Setting CurCostNoLambda to %f\n",CurCostNoLambda);
		setCurRegulaizerCostNoLambda(CurCostNoLambda);
	}
			
    PTR_FREE(SumS);
}
