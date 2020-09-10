/* Copyright 2015-2017. The Regents of the University of California.
 * Copyright 2015-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2015-2016 Frank Ong <frankong@berkeley.edu>
 * 2015-2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 *
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "num/multind.h"
#include "num/iovec.h"
#include "num/ops.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"
#include "linops/grad.h"
#include "linops/sum.h"
#include "linops/waveop.h"

#include "wavelet/wavthresh.h"

#include "lowrank/lrthresh.h"

#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "optreg.h"

#include "linops/linopScript.h"

#define CFL_SIZE sizeof(complex float)

const struct linop_s* HankelLinop=NULL;
void SetLinop(const struct linop_s* L) {HankelLinop=L;};


void help_reg(void)
{
	printf( "Generalized regularization options (experimental)\n\n"
			"-R <T>:A:B:C\t<T> is regularization type (single letter),\n"
			"\t\tA is transform flags, B is joint threshold flags,\n"
			"\t\tand C is regularization value. Specify any number\n"
			"\t\tof regularization terms.\n\n"
			"-R Q:C    \tl2-norm in image domain\n"
			"-R I:B:C  \tl1-norm in image domain\n"
			"-R W:A:B:C\tl1-wavelet\n"
		    "-R N:A:B:C\tNormalized Iterative Hard Thresholding (NIHT), image domain\n"
		        "\t\tC is an integer percentage, i.e. from 0-100\n"
		        "-R H:A:B:C\tNIHT, wavelet domain\n"
			"-R F:A:B:C\tl1-Fourier\n"
			"-R T:A:B:C\ttotal variation\n"
			"-R D:A:B:C\tL2 finite differences\n"
			"-R T:7:0:.01\t3D isotropic total variation with 0.01 regularization.\n"
			"-R L:7:7:.02\tLocally low rank with spatial decimation and 0.02 regularization.\n"
			"-R M:7:7:.03\tMulti-scale low rank with spatial decimation and 0.03 regularization.\n"
			"-R K:7:7:.03:HankelizationK:BlkSize:Option:Dim\tHankelized low-rank.\n"
			"-- for linop Script\n"
			"-R 1:B:C\tl1-norm\n"
			"-R 2:C\tl2-norm\n"
			"-R 3:A:B:C:BlkSize\tl1-schatten-norm with locality (LLR)\n"
			"TV and wavelet (T,W) are also linopScript supported by chaining\n"
	      );
}




bool opt_reg(void* ptr, char c, const char* optarg)
{
	struct opt_reg_s* p = ptr;
	struct reg_s* regs = p->regs;
	const int r = p->r;
	const float lambda = p->lambda;

	assert(r < NUM_REGS);

	char rt[5];

	switch (c) {

	case 'R': {

		// first get transform type
		int ret = sscanf(optarg, "%4[^:]", rt);
		assert(1 == ret);

		// next switch based on transform type
		if (strcmp(rt, "W") == 0) {

			regs[r].xform = L1WAV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "H") == 0) {
			
			regs[r].xform = NIHTWAV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%d", &regs[r].xflags, &regs[r].jflags, &regs[r].k);
			assert(3 == ret);
		}
		else if (strcmp(rt, "N") == 0) {
			
			regs[r].xform = NIHTIM;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%d", &regs[r].xflags, &regs[r].jflags, &regs[r].k);
			assert(3 == ret);
		}
		else if (strcmp(rt, "L") == 0) {

			regs[r].xform = LLR;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "K") == 0) {

			regs[r].xform = HLLR;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f:%d:%d:%d:%d", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda,&regs[r].k,&regs[r].q,&regs[r].k2,&regs[r].k3);
			assert(7 == ret);
		}
		else if (strcmp(rt, "1") == 0) {

			regs[r].xform = L1LS;
			int ret = sscanf(optarg, "%*[^:]:%d:%f", &regs[r].jflags, &regs[r].lambda);
			assert(2 == ret);
		}
		else if (strcmp(rt, "2") == 0) {

			regs[r].xform = L2LS;
			int ret = sscanf(optarg, "%*[^:]:%f:%d", &regs[r].lambda,&regs[r].k2);
			assert( (1 == ret) || (ret == 2) );
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;
			if(ret==1) { regs[r].k2=-1; }
		}
		else if (strcmp(rt, "3") == 0) {

			regs[r].xform = LRLS;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f:%d", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda,&regs[r].k);
			assert(4 == ret);
		}
		else if (strcmp(rt, "M") == 0) {

			// FIXME: here an explanation is missing

			regs[r].xform = regs[0].xform;
			regs[r].xflags = regs[0].xflags;
			regs[r].jflags = regs[0].jflags;
			regs[r].lambda = regs[0].lambda;

			regs[0].xform = MLR;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[0].xflags, &regs[0].jflags, &regs[0].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "T") == 0) {

			regs[r].xform = TV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "D") == 0) {

			regs[r].xform = L2FD;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "P") == 0) {

			regs[r].xform = LAPLACE;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "R1") == 0) {

			regs[r].xform = IMAGL1;
			int ret = sscanf(optarg, "%*[^:]:%d:%f", &regs[r].jflags, &regs[r].lambda);
			assert(2 == ret);
			regs[r].xflags = 0u;
		}
		else if (strcmp(rt, "R2") == 0) {

			regs[r].xform = IMAGL2;
			int ret = sscanf(optarg, "%*[^:]:%d:%f", &regs[r].jflags, &regs[r].lambda);
			assert(2 == ret);
			regs[r].xflags = 0u;
		}
		else if (strcmp(rt, "I") == 0) {

			regs[r].xform = L1IMG;
			int ret = sscanf(optarg, "%*[^:]:%d:%f", &regs[r].jflags, &regs[r].lambda);
			assert(2 == ret);
			regs[r].xflags = 0u;
		}
		else if (strcmp(rt, "S") == 0) {

			regs[r].xform = POS;
			regs[r].lambda = 0u;
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;
		}
		else if (strcmp(rt, "Q") == 0) {

			regs[r].xform = L2IMG;
			int ret = sscanf(optarg, "%*[^:]:%f", &regs[r].lambda);
			assert(1 == ret);
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;
		}
		else if (strcmp(rt, "C") == 0) {

			regs[r].xform = L2CH;
			int ret = sscanf(optarg, "%*[^:]:%f", &regs[r].lambda);
			assert(1 == ret);
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;
		}
		else if (strcmp(rt, "F") == 0) {

			regs[r].xform = FTL1;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "h") == 0) {

			help_reg();
			exit(0);
		}
		else {

			error("Unrecognized regularization type: \"%s\" (-Rh for help).\n", rt);
		}

		p->r++;
		break;
	}

	case 'l':

		assert(r < NUM_REGS);
		regs[r].lambda = lambda;
		regs[r].xflags = 0u;
		regs[r].jflags = 0u;

		if (0 == strcmp("1", optarg)) {

			regs[r].xform = L1WAV;
			regs[r].xflags = 7u;

		} else if (0 == strcmp("2", optarg)) {

			regs[r].xform = L2IMG;

		} else {

			error("Unknown regularization type.\n");
		}

		p->lambda = -1.;
		p->r++;
		break;
	}

	return false;
}

bool opt_reg_init(struct opt_reg_s* ropts)
{
	ropts->r = 0;
	ropts->lambda = -1;
	ropts->k = 0;

	return false;
}


void opt_bpursuit_configure(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], const struct linop_s* model_op, const complex float* data, const float eps)
{
	int nr_penalties = ropts->r;
	assert(NUM_REGS > nr_penalties);

	const struct iovec_s* iov = linop_codomain(model_op);
	prox_ops[nr_penalties] = prox_l2ball_create(iov->N, iov->dims, eps, data);
	trafos[nr_penalties] = linop_clone(model_op);

	ropts->r++;
}

void opt_reg_configure(unsigned int N, const long img_dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS],
	const struct linop_s* trafos[NUM_REGS], unsigned int llr_blk, unsigned int shift_mode, bool use_gpu)
{
	float lambda = ropts->lambda;
	bool randshift = shift_mode == 1;
	bool overlapping_blocks = shift_mode == 2;

	if (-1. == lambda)
		lambda = 0.;

	// if no penalities specified but regularization
	// parameter is given, add a l2 penalty

	struct reg_s* regs = ropts->regs;

	if ((0 == ropts->r) && (lambda > 0.)) {

		regs[0].xform = L2IMG;
		regs[0].xflags = 0u;
		regs[0].jflags = 0u;
		regs[0].lambda = lambda;
		ropts->r = 1;
	}



	int nr_penalties = ropts->r;
	long blkdims[MAX_LEV][DIMS];
	int levels;


	for (int nr = 0; nr < nr_penalties; nr++) {
		// ggg linopScript
		debug_printf(DP_INFO, "---- Regularizer #%d -------\n",nr);
		debug_printf(DP_INFO, "---- Linop out counter: %d -------\n",getLinopOutCounter());
		
		// fix up regularization parameter
		if (-1. == regs[nr].lambda)
			regs[nr].lambda = lambda;

		switch (regs[nr].xform) {

		case L1WAV:
		{

			debug_printf(DP_INFO, "l1-wavelet regularization: %f randshift %d\n", regs[nr].lambda,randshift);


			long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
			minsize[0] = MIN(img_dims[0], 16);
			minsize[1] = MIN(img_dims[1], 16);
			minsize[2] = MIN(img_dims[2], 16);


			unsigned int wflags = 0;
			for (unsigned int i = 0; i < DIMS; i++) {

				if ((1 < img_dims[i]) && MD_IS_SET(regs[nr].xflags, i)) {

					wflags = MD_SET(wflags, i);
					minsize[i] = MIN(img_dims[i], 16);
				}
			}

			if(nr>=getLinopOutCounter()-1) {
				trafos[nr] = linop_identity_create(DIMS, img_dims);
			} else {
				debug_printf(DP_INFO, "Using linopScript\n");
				trafos[nr] = getLinopsOutVec()[nr+1];
			}
			//prox_ops[nr] = prox_wavelet_thresh_create(DIMS, img_dims, wflags, regs[nr].jflags, minsize, regs[nr].lambda, randshift);
			prox_ops[nr] = prox_wavelet_thresh_create(DIMS, linop_codomain(trafos[nr])->dims, wflags, regs[nr].jflags, minsize,
														regs[nr].lambda, randshift);
			break;
		}
		
		case NIHTWAV:
		{
			debug_printf(DP_INFO, "NIHT with wavelets regularization: k = %d%% of total elements in each wavelet transform\n", regs[nr].k);

			if (use_gpu)
				error("GPU operation is not currently implemented for NIHT.\n");

			long img_strs[N];
			md_calc_strides(N, img_strs, img_dims, CFL_SIZE);

			long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
			minsize[0] = MIN(img_dims[0], 16);
			minsize[1] = MIN(img_dims[1], 16);
			minsize[2] = MIN(img_dims[2], 16);


			unsigned int wflags = 0;
			unsigned int wxdim = 0;
			for (unsigned int i = 0; i < DIMS; i++) {

				if ((1 < img_dims[i]) && MD_IS_SET(regs[nr].xflags, i)) {

					wflags = MD_SET(wflags, i);
					minsize[i] = MIN(img_dims[i], 16);
					wxdim += 1;
				}
			}

			trafos[nr] = linop_wavelet_create(N, wflags, img_dims, img_strs, minsize, randshift);

			long wav_dims[DIMS];
			md_copy_dims(DIMS, wav_dims, linop_codomain(trafos[nr])->dims);
			unsigned int K = (md_calc_size(wxdim, wav_dims) / 100) * regs[nr].k;

			debug_printf(DP_DEBUG3, "\nK = %d elements will be thresholded per wavelet transform\n", K);
			debug_printf(DP_DEBUG3, "Total wavelet dimensions: \n[");
			for (unsigned int i = 0; i < DIMS; i++)
				debug_printf(DP_DEBUG3,"%d ", wav_dims[i]);
			debug_printf(DP_DEBUG3, "]\n");
			
			prox_ops[nr] = prox_niht_thresh_create(N, wav_dims, K, regs[nr].jflags);
			break;
		}

		case NIHTIM:
		{
			debug_printf(DP_INFO, "NIHT regularization in the image domain: k = %d%% of total elements in image vector\n", regs[nr].k);

			if (use_gpu)
				error("GPU operation is not currently implemented for NIHT.\n");

			long thresh_dims[N];
			md_select_dims(N, regs[nr].xflags, thresh_dims, img_dims);		
			unsigned int K = (md_calc_size(N, thresh_dims) / 100) * regs[nr].k;
			debug_printf(DP_INFO, "k = %d%%, actual K = %d\n", regs[nr].k, K);

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			prox_ops[nr] = prox_niht_thresh_create(N, img_dims, K, regs[nr].jflags);
			debug_printf(DP_INFO, "NIHTIM initialization complete\n");
			break;
		}

		case TV:
			debug_printf(DP_INFO, "TV regularization: %f\n", regs[nr].lambda);

			/// ggg LinopScript
			if(nr<getLinopOutCounter()-1) {
				const struct linop_s* tmp=linop_grad_create(linop_codomain(getLinopsOutVec()[nr+1])->N, linop_codomain(getLinopsOutVec()[nr+1])->dims, regs[nr].xflags);
				trafos[nr] = linop_chain(getLinopsOutVec()[nr+1],tmp);
				linop_free(tmp);
			} else {
				trafos[nr] = linop_grad_create(DIMS, img_dims, regs[nr].xflags);
			}
			
			prox_ops[nr] = prox_thresh_create(DIMS + 1,
					linop_codomain(trafos[nr])->dims,
					regs[nr].lambda, regs[nr].jflags | MD_BIT(DIMS));
			break;

		case L2FD:
			debug_printf(DP_INFO, "L2FD regularization: %f\n", regs[nr].lambda);

			/// ggg LinopScript
			if(nr<getLinopOutCounter()-1) {
				const struct linop_s* tmp=linop_grad_create(linop_codomain(getLinopsOutVec()[nr+1])->N, linop_codomain(getLinopsOutVec()[nr+1])->dims, regs[nr].xflags);
				trafos[nr] = linop_chain(getLinopsOutVec()[nr+1],tmp);
				linop_free(tmp);
			} else {
				trafos[nr] = linop_grad_create(DIMS, img_dims, regs[nr].xflags);
			}
			
			/*prox_ops[nr] = prox_thresh_create(DIMS + 1,
					linop_codomain(trafos[nr])->dims,
					regs[nr].lambda, regs[nr].jflags | MD_BIT(DIMS));*/

			prox_ops[nr] = prox_leastsquares_create(DIMS + 1,
							linop_codomain(trafos[nr])->dims, regs[nr].lambda, NULL);
			break;

		case L1LS:
			debug_printf(DP_INFO, "L1 regularization (for linopScript): %f\n", regs[nr].lambda);

			// trafos[nr] = linop_grad_create(DIMS, img_dims, regs[nr].xflags);

			if(nr>=getLinopOutCounter()-1) {
				trafos[nr] = linop_identity_create(DIMS, img_dims);
			} else {
				trafos[nr] = getLinopsOutVec()[nr+1];
			}
			
			prox_ops[nr] = prox_thresh_create(linop_codomain(trafos[nr])->N,
					linop_codomain(trafos[nr])->dims,
					regs[nr].lambda, regs[nr].jflags);
			break;

		case L2LS:
			debug_printf(DP_INFO, "L2 regularization (for linopScript): %f\n", regs[nr].lambda);

			trafos[nr] = linop_grad_create(DIMS, img_dims, regs[nr].xflags);

			if(nr>=getLinopOutCounter()) {
				trafos[nr] = linop_identity_create(DIMS, img_dims);
			} else {
				trafos[nr] = getLinopsOutVec()[nr+1];
			}

			complex float* Fdata=NULL;
			if(regs[nr].k2>=0) {
				debug_printf(DP_INFO, "Using file %d !!!\n", regs[nr].k2);
				Fdata=getDataFile(regs[nr].k2);
			}

			prox_ops[nr] = prox_leastsquares_create(linop_codomain(trafos[nr])->N,
							linop_codomain(trafos[nr])->dims, regs[nr].lambda, Fdata);
			break;

		case LRLS:
			debug_printf(DP_INFO, "LR regularization (for linopScript): %f\n", regs[nr].lambda);

			if (use_gpu)
				error("GPU operation is not currently implemented for lowrank regularization.\n");

			if(nr>=getLinopOutCounter()) {
				debug_printf(DP_INFO, "NO linop script for this one!\n");
				trafos[nr] = linop_identity_create(DIMS, img_dims);
			} else {
				debug_printf(DP_INFO, "Using linop script #%d\n",nr+1);
				trafos[nr] = getLinopsOutVec()[nr+1];
			}

			// add locally lowrank penalty
			levels = llr_blkdims(blkdims, regs[nr].jflags, linop_codomain(trafos[nr])->dims, regs[nr].k);

			assert(1 == levels);

			assert(levels == img_dims[LEVEL_DIM]);

			for(int l = 0; l < levels; l++)
#if 0
				blkdims[l][MAPS_DIM] = img_dims[MAPS_DIM];
#else
				blkdims[l][MAPS_DIM] = 1;
#endif

			int LSremove_mean = 0;

			prox_ops[nr] = lrthresh_create(linop_codomain(trafos[nr])->dims, randshift, regs[nr].xflags,
							(const long (*)[DIMS])blkdims, regs[nr].lambda, false, LSremove_mean, overlapping_blocks);
			break;

		case LAPLACE:
			debug_printf(DP_INFO, "L1-Laplace regularization: %f\n", regs[nr].lambda);
			long krn_dims[DIMS] = { [0 ... DIMS - 1] = 1 };

			for (unsigned int i = 0; i < DIMS; i++)
				if (MD_IS_SET(regs[nr].xflags, i))
					krn_dims[i] = 3;

			complex float krn[] = {	// laplace filter
				-1., -2., -1.,
				-2., 12., -2.,
				-1., -2., -1.,
			};

			assert(9 == md_calc_size(DIMS, krn_dims));

			trafos[nr] = linop_conv_create(DIMS, regs[nr].xflags, CONV_TRUNCATED, CONV_SYMMETRIC, img_dims, img_dims, krn_dims, krn);
			prox_ops[nr] = prox_thresh_create(DIMS,
					linop_codomain(trafos[nr])->dims,
					regs[nr].lambda, regs[nr].jflags);
			break;

		case LLR:

			debug_printf(DP_INFO, "lowrank regularization: %f\n", regs[nr].lambda);

			// ggg
			// if (use_gpu)
			// 	error("GPU operation is not currently implemented for lowrank regularization.\n");


			// add locally lowrank penalty
			levels = llr_blkdims(blkdims, regs[nr].jflags, img_dims, llr_blk);

			assert(1 == levels);

			assert(levels == img_dims[LEVEL_DIM]);

			for(int l = 0; l < levels; l++)
#if 0
				blkdims[l][MAPS_DIM] = img_dims[MAPS_DIM];
#else
				blkdims[l][MAPS_DIM] = 1;
#endif

			int remove_mean = 0;

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			prox_ops[nr] = lrthresh_create(img_dims, randshift, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda, false, remove_mean, overlapping_blocks);
			break;

		case HLLR:

			debug_printf(DP_INFO, "lowrank regularization: %f TH %d llrBlock %d\n", regs[nr].lambda,regs[nr].k,regs[nr].q);

			// if (use_gpu)
			// 	error("GPU operation is not currently implemented for lowrank regularization.\n");


			// ggg changed to control via flags
			// add locally lowrank penalty
			// levels = llr_blkdims(blkdims, regs[nr].jflags, img_dims, llr_blk);
			levels = llr_blkdims(blkdims, regs[nr].jflags, img_dims, regs[nr].q);

			assert(1 == levels);

			assert(levels == img_dims[LEVEL_DIM]);

			for(int l = 0; l < levels; l++)
#if 0
				blkdims[l][MAPS_DIM] = img_dims[MAPS_DIM];
#else
				blkdims[l][MAPS_DIM] = 1;
#endif

			int Hremove_mean = 0;

			// trafos[nr] = linop_identity_create(DIMS, img_dims);
			long HDim1=regs[nr].k3;
			long HDim2=HDim1+1;
			// long HankelK=2;
			trafos[nr] = linop_Hankel_create(DIMS, img_dims,HDim1,HDim2,regs[nr].k);

			if(randshift) {
				debug_printf(DP_INFO, "randshift true\n");
			} else {
				debug_printf(DP_INFO, "randshift false\n");
			}

			debug_printf(DP_INFO, "trafos codims: ");
			debug_print_dims(DP_INFO,DIMS,linop_codomain(trafos[nr])->dims);

			blkdims[0][HDim1]=linop_codomain(trafos[nr])->dims[HDim1];
			blkdims[0][HDim2]=linop_codomain(trafos[nr])->dims[HDim2];

			prox_ops[nr] = lrthresh_createx(linop_codomain(trafos[nr])->dims, randshift, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda,
				false, Hremove_mean, overlapping_blocks,regs[nr].k2);
			break;
		
// 		case HLLR:

// 			debug_printf(DP_INFO, "Hankelized lowrank regularization: %f\n", regs[nr].lambda);

// 			if (use_gpu)
// 				error("GPU operation is not currently implemented for Hankelized lowrank regularization.\n");

// 			if(randshift) {
// 				debug_printf(DP_INFO, "randshift true\n");
// 			} else {
// 				debug_printf(DP_INFO, "randshift false\n");
// 			}
			

// 			// add locally lowrank penalty
// 			levels = llr_blkdims(blkdims, regs[nr].jflags, img_dims, llr_blk);

// 			assert(1 == levels);

// 			assert(levels == img_dims[LEVEL_DIM]);

// // 			for(int l = 0; l < levels; l++)
// // #if 0
// // 				blkdims[l][MAPS_DIM] = img_dims[MAPS_DIM];
// // #else
// // 				blkdims[l][MAPS_DIM] = 1;
// // #endif


// // trafos[nr] = linop_grad_create(DIMS, img_dims, regs[nr].xflags);
// // 			prox_ops[nr] = prox_thresh_create(DIMS + 1,
// // 					linop_codomain(trafos[nr])->dims,
// // 					regs[nr].lambda, regs[nr].jflags | MD_BIT(DIMS));

// 			int remove_meanx = 0;

			

			
// 			blkdims[0][5]=1;
// 			blkdims[0][0]=1;
// 			blkdims[0][1]=1;
// 			blkdims[0][6]=6;
// 			blkdims[0][7]=3;
			
                
// //                 // printf("Flags: %ld %ld %ld\n",CurFlags,NewFlags,TFlags);
			

// 			// trafos[nr] = linop_print_create(DIMS, img_dims,77);
// 			trafos[nr] = HankelLinop;

// 			debug_printf(DP_INFO, "trafos codims: ");
// 			debug_print_dims(DP_INFO,DIMS,linop_codomain(trafos[nr])->dims);
// 			debug_printf(DP_INFO, "Creating HLLR_ThreshOp\n");
// 			const struct operator_p_s* HLLR_ThreshOp=lrthresh_create(linop_codomain(trafos[nr])->dims, 
// 				/*randshift*/ false, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda, false, remove_meanx, overlapping_blocks);

// 			/*const struct operator_p_s* ThreshWithH=operator_chain(HankelForw,HLLR_ThreshOp);
// 			const struct operator_p_s* ThreshWithHandDeH=operator_chain(ThreshWithH,DeHankelForw);*/

// // 			MergedDims;
// // 			unsigned int oflags;
// // 			unsigned int iflags;
// // 			unsigned int tflags;
// // linop_fmac_create(DIMS, MergedDims, 
// // 		unsigned int oflags, unsigned int iflags, unsigned int tflags, const complex float* tensor)


// 			// extern const struct operator_s* operator_chain(const struct operator_s* a, const struct operator_s* b);

// 			//trafos[nr] = linop_print_create(DIMS, img_dims,77);
// 			// prox_ops[nr] = lrthresh_create(img_dims, randshift, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda, false, remove_meanx, overlapping_blocks);
// 			prox_ops[nr] = HLLR_ThreshOp;
// 			//operator_p_free()
// 			break;

		case MLR:
#if 0
			// FIXME: multiscale low rank changes the output image dimensions 
			// and requires the forward linear operator. This should be decoupled...
			debug_printf(DP_INFO, "multi-scale lowrank regularization: %f\n", regs[nr].lambda);

			levels = multilr_blkdims(blkdims, regs[nr].jflags, img_dims, 8, 1);

			img_dims[LEVEL_DIM] = levels;
			max_dims[LEVEL_DIM] = levels;

			for(int l = 0; l < levels; l++)
				blkdims[l][MAPS_DIM] = 1;

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			prox_ops[nr] = lrthresh_create(img_dims, randshift, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda, false, 0, use_gpu);

			const struct linop_s* decom_op = sum_create( img_dims, use_gpu );
			const struct linop_s* tmp_op = forward_op;
			forward_op = linop_chain(decom_op, forward_op);

			linop_free(decom_op);
			linop_free(tmp_op);
#else
			error("multi-scale lowrank regularization not supported.\n");
#endif

			break;

		case IMAGL1:
			debug_printf(DP_INFO, "l1 regularization of imaginary part: %f\n", regs[nr].lambda);

			trafos[nr] = linop_rdiag_create(DIMS, img_dims, 0, &(complex float){ 1.i });
			prox_ops[nr] = prox_thresh_create(DIMS, img_dims, regs[nr].lambda, regs[nr].jflags);
			break;

		case IMAGL2:
			debug_printf(DP_INFO, "l2 regularization of imaginary part: %f\n", regs[nr].lambda);

			trafos[nr] = linop_rdiag_create(DIMS, img_dims, 0, &(complex float){ 1.i });
			prox_ops[nr] = prox_leastsquares_create(DIMS, img_dims, regs[nr].lambda, NULL);
			break;

		case L1IMG:
			debug_printf(DP_INFO, "l1 regularization: %f\n", regs[nr].lambda);

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			prox_ops[nr] = prox_thresh_create(DIMS, img_dims, regs[nr].lambda, regs[nr].jflags);
			break;

		case POS:
			debug_printf(DP_INFO, "non-negative constraint\n");

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			prox_ops[nr] = prox_nonneg_create(DIMS, img_dims);
			break;

		case L2CH:
		{
			debug_printf(DP_INFO, "l2 channels: %f\n", regs[nr].lambda);

			complex float* MMI;
			long MMI_dims[DIMS];

			MMI = load_cfl("/autofs/space/daisy_002/users/Gilad/gUM/MMI", DIMS, MMI_dims);

			long CurDims[DIMS];
			md_copy_dims(DIMS, CurDims, img_dims);

			debug_printf(DP_INFO,"MMI_dims : ");
            debug_print_dims(DP_INFO,DIMS,MMI_dims);

			debug_printf(DP_INFO,"CurDims : ");
            debug_print_dims(DP_INFO,DIMS,CurDims);

			long SquashFlags=4;

			long MergedDims[DIMS];
            md_merge_dims(DIMS, MergedDims, CurDims, MMI_dims);
            long NewDims[DIMS];
            md_select_dims(DIMS, ~SquashFlags, NewDims, MergedDims);
            
			debug_printf(DP_INFO,"MergedDims : ");
            debug_print_dims(DP_INFO,DIMS,MergedDims);

			debug_printf(DP_INFO,"NewDims : ");
            debug_print_dims(DP_INFO,DIMS,NewDims);

			long CurFlags=md_nontriv_dims(DIMS,CurDims);
			long NewFlags=md_nontriv_dims(DIMS,NewDims);
			long TFlags=md_nontriv_dims(DIMS,MMI_dims);

			// complex float* ET36;
			// long ET36_dims[DIMS];

			// ET36 = load_cfl("/autofs/space/daisy_002/users/Gilad/gUM/ET36", DIMS, ET36_dims);

			// long MergedDims2[DIMS];

			// long CurFlags2=md_nontriv_dims(DIMS,CurDims);
			// long NewFlags2=md_nontriv_dims(DIMS,NewDims);
			// long TFlags2=md_nontriv_dims(DIMS,MMI_dims);

			// const struct linop_s* fmacop2=linop_fmac_create(DIMS, MergedDims, ~CurFlags, ~NewFlags, ~TFlags, ET36);

			const struct linop_s* fmacop=linop_fmac_create(DIMS, MergedDims, ~NewFlags, ~CurFlags, ~TFlags, MMI);
			// const struct linop_s* fmacop=linop_Hankel_create(DIMS, img_dims,2,6,2);

			const struct linop_s* pp=linop_print_create(DIMS, CurDims, 33);
			const struct linop_s* pp2=linop_print_create(DIMS, linop_codomain(fmacop)->dims, 44);
			// const struct linop_s* pp3=linop_print_create(DIMS, CurDims, 55);
			
			pp = linop_chain(pp,fmacop);
			pp = linop_chain(pp,pp2);
			// pp = linop_chain(pp,fmacop2);
			// pp = linop_chain(pp,pp3);

			// const struct linop_s* top=linop_transpose_create(DIMS, CurDims, 3,6);
			// pp = linop_chain(pp,top);
			// pp = linop_chain(pp,pp3);

            trafos[nr] = pp;
			// trafos[nr] = linop_Hankel_create(DIMS, img_dims,2,6,2);
			// trafos[nr] =linop_transpose_create(DIMS, CurDims, 3,6);
			// trafos[nr] =linop_fmac_create(DIMS, MergedDims, ~NewFlags, ~CurFlags, ~TFlags, MMI);

			// prox_ops[nr] = prox_leastsquares_create(DIMS, NewDims, regs[nr].lambda, NULL);
			prox_ops[nr] = prox_leastsquares_create(DIMS, linop_codomain(trafos[nr])->dims, regs[nr].lambda, NULL);

			// trafos[nr] = linop_identity_create(DIMS, img_dims);
			// Do fmac!!!!!!!!!! with CH'*Ch-I
			
			// prox_ops[nr] = prox_leastsquares_create(DIMS, img_dims, regs[nr].lambda, NULL);
			break;
		}

		case L2IMG:
			debug_printf(DP_INFO, "l2 regularization: %f\n", regs[nr].lambda);

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			prox_ops[nr] = prox_leastsquares_create(DIMS, img_dims, regs[nr].lambda, NULL);
			break;

		case FTL1:
			debug_printf(DP_INFO, "l1 regularization of Fourier transform: %f\n", regs[nr].lambda);

			trafos[nr] = linop_fft_create(DIMS, img_dims, regs[nr].xflags);
			prox_ops[nr] = prox_thresh_create(DIMS, img_dims, regs[nr].lambda, regs[nr].jflags);
			break;
		}

	}
}


void opt_reg_free(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS])
{
	int nr_penalties = ropts->r;

	for (int nr = 0; nr < nr_penalties; nr++) {

		operator_p_free(prox_ops[nr]);
		linop_free(trafos[nr]);
	}
}
