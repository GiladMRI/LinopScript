/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2015-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014-2016 Frank Ong <frankong@berkeley.edu>
 * 2014-2018 Jon Tamir <jtamir@eecs.berkeley.edu>
 *
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops.h"

#include "iter/misc.h"

#include "linops/linop.h"
#include "linops/linopScript.h"
#include "linops/fmac.h"
#include "linops/sampling.h"
#include "linops/someops.h"

#include "noncart/nufft.h"

#include "sense/recon.h"
#include "sense/model.h"
#include "sense/optcom.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#include "num/iovec.h"
#include "num/ops.h"

static const char usage_str[] = "<output> <kspace> <sensitivities> [<input 2> [<input3> [...]]]";
static const char help_str[] = "Parallel-imaging compressed-sensing reconstruction - linop script.";

int main_picsS(int argc, char* argv[])
{
	// Initialize default parameters

	struct sense_conf conf = sense_defaults;

	float bpsense_eps = -1.;

	unsigned int shift_mode = 0;
	bool randshift = true;
	bool overlapping_blocks = false;
	unsigned int maxiter = 30;
	float step = -1.;

	// Start time count

	double start_time = timestamp();

    // Linop script added here
    const char* normalScript_file = NULL;
    const char* normalScriptForward_file = NULL;
	unsigned int fftmod_flags = 7;
    
	// Read input options
	struct nufft_conf_s nuconf = nufft_conf_defaults;
	nuconf.toeplitz = true;
	nuconf.lowmem = true;

	float restrict_fov = -1.;
	bool scale_im = false;
	bool eigen = false;
	float scaling = 0.;

	unsigned int llr_blk = 8;

	const char* image_truth_file = NULL;
	bool im_truth = false;

	const char* image_start_file = NULL;
	bool warm_start = false;

	struct admm_conf admm = { false, false, false, iter_admm_defaults.rho, iter_admm_defaults.maxitercg };

	enum algo_t algo = ALGO_DEFAULT;

	bool hogwild = false;
	bool fast = false;

	unsigned int gpun = 0;

	struct opt_reg_s ropts;
	opt_reg_init(&ropts);

	unsigned int loop_flags = 0u;

	const struct opt_s opts[] = {

		{ 'l', true, opt_reg, &ropts, "1/-l2\t\ttoggle l1-wavelet or l2 regularization." },
		OPT_FLOAT('r', &ropts.lambda, "lambda", "regularization parameter"),
		{ 'R', true, opt_reg, &ropts, " <T>:A:B:C\tgeneralized regularization options (-Rh for help)" },
		OPT_SET('c', &conf.rvc, "real-value constraint"),
		OPT_FLOAT('s', &step, "step", "iteration stepsize"),
		OPT_UINT('i', &maxiter, "iter", "max. number of iterations"),
		OPT_CLEAR('n', &randshift, "disable random wavelet cycle spinning"),
		OPT_SET('N', &overlapping_blocks, "do fully overlapping LLR blocks"),
		OPT_SET('g', &conf.gpu, "use GPU"),
		OPT_UINT('G', &gpun, "gpun", "use GPU device gpun"),
		OPT_SELECT('I', enum algo_t, &algo, ALGO_IST, "select IST"),
		OPT_UINT('b', &llr_blk, "blk", "Lowrank block size"),
		OPT_SET('e', &eigen, "Scale stepsize based on max. eigenvalue"),
		OPT_SET('H', &hogwild, "(hogwild)"),
		OPT_SET('D', &admm.dynamic_rho, "(ADMM dynamic step size)"),
		OPT_SET('F', &fast, "(fast)"),
		OPT_SET('J', &admm.relative_norm, "(ADMM residual balancing)"),
		OPT_STRING('T', &image_truth_file, "file", "(truth file)"),
		OPT_STRING('W', &image_start_file, "<img>", "Warm start with <img>"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_INT('O', &conf.rwiter, "rwiter", "(reweighting)"),
		OPT_FLOAT('o', &conf.gamma, "gamma", "(reweighting)"),
		OPT_FLOAT('u', &admm.rho, "rho", "ADMM rho"),
		OPT_UINT('C', &admm.maxitercg, "iter", "ADMM max. CG iterations"),
		OPT_FLOAT('q', &conf.cclambda, "cclambda", "(cclambda)"),
		OPT_FLOAT('f', &restrict_fov, "rfov", "restrict FOV"),
		OPT_SELECT('m', enum algo_t, &algo, ALGO_ADMM, "select ADMM"),
		OPT_FLOAT('w', &scaling, "val", "inverse scaling of the data"),
		OPT_SET('S', &scale_im, "re-scale the image after reconstruction"),
		OPT_UINT('L', &loop_flags, "flags", "batch-mode"),
		OPT_SET('K', &nuconf.pcycle, "randshift for NUFFT"),
		OPT_FLOAT('P', &bpsense_eps, "eps", "Basis Pursuit formulation, || y- Ax ||_2 <= eps"),
		OPT_SELECT('a', enum algo_t, &algo, ALGO_PRIDU, "select Primal Dual"),
        // Linop script added here
        OPT_STRING('M', &normalScript_file, "file", "script for Normal operator"),
        OPT_STRING('Q', &normalScriptForward_file, "file", "script for forward part of Normal operator"),
        OPT_UINT('j', &fftmod_flags, "fftmod_flags", "flags for fftmod_flags of k-space input and sensitivity maps"),
	};

	cmdline(&argc, argv, 3, 100, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (NULL != image_truth_file)
		im_truth = true;

	if (NULL != image_start_file)
		warm_start = true;

	if (0 <= bpsense_eps)
		conf.bpsense = true;

	admm.dynamic_tau = admm.relative_norm;

	if (conf.bpsense)
		nuconf.toeplitz = false;


	long max_dims[DIMS];
    // Linop script: removed
	//long map_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	// Linop script: removed
	// long coilim_dims[DIMS];
	long ksp_dims[DIMS];

    long img_start_dims[DIMS];
	complex float* image_start = NULL;

    // Linop script :  moved reading warm start to the beginning
	if (warm_start) { 

		debug_printf(DP_DEBUG1, "Warm start: %s\n", image_start_file);

		image_start = load_cfl(image_start_file, DIMS, img_start_dims);

        // Linop script: changed this
// 		assert(md_check_compat(DIMS, 0u, img_start_dims, img_dims));
        md_select_dims(DIMS, ~0, img_dims, img_start_dims);
        debug_printf(DP_INFO, "Warm start: Starting with ");
        debug_print_dims(DP_INFO,DIMS,img_start_dims);

		xfree(image_start_file);
    }


	
//     debug_printf(DP_INFO, "img_dims");
// 	debug_print_dims(DP_INFO,DIMS,img_dims);
    
    // load kspace and maps and get dimensions
    // Linop script: changed to argv[3]
	complex float* kspace = load_cfl(argv[3], DIMS, ksp_dims);

    // Linop script: removed. ## changed to argv[4]
// 	complex float* maps = load_cfl(argv[4], DIMS, map_dims);
// 	unsigned int map_flags = md_nontriv_dims(DIMS, map_dims);
// 	map_flags |= FFT_FLAGS | SENS_FLAGS;



	md_copy_dims(DIMS, max_dims, ksp_dims);
    // Linop script: removed
// 	md_copy_dims(5, max_dims, map_dims);

	// Linop script: removed
	// assert(1 == ksp_dims[COEFF_DIM]);
	
    // Linop script: removed
// 	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);
	// md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);

    // Linop script: removed
// 	if (!md_check_compat(DIMS, ~(MD_BIT(MAPS_DIM)|FFT_FLAGS), img_dims, map_dims))
// 		error("Dimensions of image and sensitivities do not match!\n");

	// Linop script: removed
	// assert(1 == ksp_dims[MAPS_DIM]);
    
    
    // Linop script
    int num_args = argc - 1;
    
	printf("main_picsS\n");
    
    ReadScriptFiles(&argv[3],num_args-2);
    
    fftmod(DIMS, getFdims(1), fftmod_flags, getDataFile(1), getDataFile(1));
            
    long CurDims[DIMS];
	md_select_dims(DIMS, ~0, CurDims, getFdims(0));
    if (warm_start) {
        md_select_dims(DIMS, ~0, CurDims, img_start_dims);    
		debug_printf(DP_INFO, "Warm start: CurDims:");
		debug_print_dims(DP_INFO,DIMS,CurDims);
	}
	debug_printf(DP_INFO, "CurDims:");
	debug_print_dims(DP_INFO,DIMS,CurDims);
    
	long dimsAfterF[DIMS];
	md_copy_dims(DIMS, dimsAfterF, CurDims);
    const struct linop_s* Sop =getLinopScriptFromFile(argv[1],dimsAfterF);
    
	debug_printf(DP_INFO, "Read forward script. dimsAfterF:");
	debug_print_dims(DP_INFO,DIMS,dimsAfterF);

    if (NULL != normalScript_file) {
		const struct linop_s* NomalSop;
            
        if(NULL != normalScriptForward_file) {
			long dimsAfterNF[DIMS];
			md_copy_dims(DIMS, dimsAfterNF, CurDims);
            const struct linop_s* NomalForwardPartSop =getLinopScriptFromFile(normalScriptForward_file,dimsAfterNF);
            
			// printf("OK linop NF script reading\n");
			// NF finishes with:
			// debug_printf(DP_INFO, "NF finishes with:");
			// debug_print_dims(DP_INFO,linop_codomain(NomalForwardPartSop)->N,linop_codomain(NomalForwardPartSop)->dims);
			// debug_printf(DP_INFO, "dimsAfterNF:");
			// debug_print_dims(DP_INFO,DIMS,dimsAfterNF);

			// md_copy_dims(DIMS, dimsAfterNF, linop_codomain(NomalForwardPartSop)->dims);
			

			// debug_printf(DP_INFO, "CurDims:");
			// debug_print_dims(DP_INFO,DIMS,CurDims);

			long dimsAfterN[DIMS];
			md_copy_dims(DIMS, dimsAfterN, dimsAfterNF);
			NomalSop =getLinopScriptFromFile(normalScript_file,dimsAfterN);
			const struct linop_s* tmp = NomalSop;
        	NomalSop = linop_PutFowrardOfBInNormalOfA(tmp,tmp);
        	linop_free(tmp);
			// printf("OK linop N script reading\n");

            const struct linop_s* tmp2 = NomalSop;
			// printf("NF chain:\n");
            NomalSop = linop_chain(NomalForwardPartSop,tmp2);
            linop_free(tmp2);
            linop_free(NomalForwardPartSop);
        } else {
			long dimsAfterN[DIMS];
			md_copy_dims(DIMS, dimsAfterN, CurDims);
			NomalSop =getLinopScriptFromFile(normalScript_file,dimsAfterN);

			const struct linop_s* tmp = NomalSop;
        	NomalSop = linop_PutFowrardOfBInNormalOfA(tmp,tmp);
        	linop_free(tmp);
			// printf("OK linop N script reading\n");
		}

		const struct linop_s* tmp3 = Sop;
		// printf("Put Normal inside op\n");
        Sop = linop_PutNormalOfBInNormalOfA(tmp3,NomalSop);
        
        linop_free(NomalSop);
        linop_free(tmp3);

		// printf("OK Normal script\n");
    }
	printf("OK linop script reading\n");

	debug_printf(DP_INFO, "img_dims:");
	debug_print_dims(DP_INFO,DIMS,img_dims);

	// Hankelization!
	long *Hankelization_dims=getFdims(3);
	long *DeHankelization_dims=getFdims(4);

	complex float* HankelizationMat = getDataFile(3);
	complex float* DeHankelizationMat = getDataFile(4);

// getFdims(long i);
// complex float* getDataFile(long i);
	debug_printf(DP_INFO, "Loading\n");

	// HankelizationMat = load_cfl("HankelizingMat", DIMS, getFdims[3]);
	// DeHankelizationMat = load_cfl("DeHankelizingMat", DIMS, getFdims[4]);

	debug_printf(DP_INFO, "Read H files\n");

	debug_printf(DP_INFO, "Hankelization_dims:");
	debug_print_dims(DP_INFO, DIMS, getFdims(3));
	debug_printf(DP_INFO, "DeHankelizationMat:");
	debug_print_dims(DP_INFO, DIMS, getFdims(4));
    // ggg Linop script
    
    
	if (conf.gpu)
		num_init_gpu_device(gpun);
	else
		num_init();

	// print options

	if (conf.gpu)
		debug_printf(DP_INFO, "GPU reconstruction\n");

    // Linop script: removed
// 	if (map_dims[MAPS_DIM] > 1) 
// 		debug_printf(DP_INFO, "%ld maps.\nESPIRiT reconstruction.\n", map_dims[MAPS_DIM]);

	if (conf.bpsense)
		debug_printf(DP_INFO, "Basis Pursuit formulation\n");

	if (hogwild)
		debug_printf(DP_INFO, "Hogwild stepsize\n");

	if (admm.dynamic_rho)
		debug_printf(DP_INFO, "ADMM Dynamic stepsize\n");

	if (admm.relative_norm)
		debug_printf(DP_INFO, "ADMM residual balancing\n");

	if (im_truth)
		debug_printf(DP_INFO, "Compare to truth\n");

	if (randshift)
		shift_mode = 1;

	if (overlapping_blocks) {

		if (randshift)
			debug_printf(DP_WARN, "Turning off random shifts\n");

		shift_mode = 2;
		debug_printf(DP_INFO, "Fully overlapping LLR blocks\n");
	}



	assert(!((conf.rwiter > 1) && (nuconf.toeplitz || conf.bpsense)));


	// print some statistics

//     long T = md_calc_size(DIMS, pat_dims);
    // long samples = (long)pow(md_znorm(DIMS, pat_dims, pattern), 2.);

    // debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T / (float)samples);
	
	//if (NULL == traj_file) {
    fftmod(DIMS, ksp_dims, fftmod_flags, kspace, kspace);
    // Linop script: removed
//     fftmod(DIMS, map_dims, FFT_FLAGS, maps, maps);
	//}

	// apply fov mask to sensitivities

    // Linop script: removed
	/*if (-1. != restrict_fov) {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;

		apply_mask(DIMS, map_dims, maps, restrict_dims);
	}*/


	// initialize forward_op and precond_op

	const struct linop_s* forward_op = NULL;
	const struct operator_s* precond_op = NULL;

    // Linop  script: changed here
// 	forward_op = sense_init(max_dims, map_flags, maps);
    forward_op = Sop;

    // apply scaling

	if (0. == scaling) {
		//if (NULL == traj_file) {
		scaling = estimate_scaling(ksp_dims, NULL, kspace);
	}

	if (0. == scaling ) {

		debug_printf(DP_WARN, "Estimated scale is zero. Set to one.");
        debug_printf(DP_INFO, "Estimated scale is zero. Set to one.");
		scaling = 1.;

	} else {

		debug_printf(DP_DEBUG1, "Inverse scaling of the data: %f\n", scaling);
        debug_printf(DP_INFO, "Inverse scaling of the data: %f\n", scaling);
		md_zsmul(DIMS, ksp_dims, kspace, kspace, 1. / scaling);

		if (conf.bpsense) {

			bpsense_eps /= scaling;
			debug_printf(DP_DEBUG1, "scaling basis pursuit eps: %.3e\n", bpsense_eps);
		}
	}

    // Linop script: changed to argv[2]
	debug_printf(DP_INFO, "Opening output file %s\n",argv[2]);
	debug_print_dims(DP_INFO,DIMS,img_dims);
	complex float* image = create_cfl(argv[2], DIMS, img_dims);
	md_clear(DIMS, img_dims, image, CFL_SIZE);
	debug_printf(DP_INFO, "Opening output file OK\n");


	long img_truth_dims[DIMS];
	complex float* image_truth = NULL;

	if (im_truth) {

		image_truth = load_cfl(image_truth_file, DIMS, img_truth_dims);
		//md_zsmul(DIMS, img_dims, image_truth, image_truth, 1. / scaling);

		xfree(image_truth_file);
	}

    // Linop script: separated and left this part here
    if (warm_start) { 
		// if rescaling at the end, assume the input has also been rescaled
		if (scale_im && (scaling != 0.))
			md_zsmul(DIMS, img_dims, image_start, image_start, 1. /  scaling);
	}



	assert((0u == loop_flags) || (NULL == image_start));
	assert(!(loop_flags & COIL_FLAG));

	const complex float* image_start1 = image_start;

	long loop_dims[DIMS];
	md_select_dims(DIMS,  loop_flags, loop_dims, max_dims);

	long img1_dims[DIMS];
	md_select_dims(DIMS, ~loop_flags, img1_dims, img_dims);

	long ksp1_dims[DIMS];
	md_select_dims(DIMS, ~loop_flags, ksp1_dims, ksp_dims);

	long max1_dims[DIMS];
	md_select_dims(DIMS, ~loop_flags, max1_dims, max_dims);

	long pat1_dims[DIMS];
	md_select_dims(DIMS, ~loop_flags, pat1_dims, pat_dims);

	// FIXME: re-initialize forward_op and precond_op

	//if ((NULL == traj_file) && (0u != loop_flags) && !sms) { // FIXME: no basis
    if ((0u != loop_flags)) { // FIXME: no basis
        debug_printf(DP_ERROR, "Linop script: This shouldn't run");
        
		linop_free(forward_op);
        // Linop script: removed this
// 		forward_op = sense_init(max1_dims, map_flags, maps);

		// basis pursuit requires the full forward model to add as a linop constraint
		// if (conf.bpsense) {

		// 	const struct linop_s* sample_op = linop_sampling_create(max1_dims, pat1_dims, pattern1);
		// 	struct linop_s* tmp = linop_chain(forward_op, sample_op);

		// 	linop_free(sample_op);
		// 	linop_free(forward_op);

		// 	forward_op = tmp;
		// }
	}

	double maxeigen = 1.;

	if (eigen) {

		maxeigen = estimate_maxeigenval(forward_op->normal);

		debug_printf(DP_INFO, "Maximum eigenvalue: %.2e\n", maxeigen);

	}


	// initialize prox functions
	printf("Preparing prox funcs\n");

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };

	

	const struct linop_s* tmpPrint = linop_print_create(DIMS, img1_dims,333);
	const struct operator_s* PrintOp=tmpPrint->forward;
	linop_free(tmpPrint);
	const struct linop_s* tmpPrint2 = linop_print_create(DIMS, img1_dims,555);
	const struct operator_s* PrintOp2=tmpPrint->forward;
	linop_free(tmpPrint2);

	debug_printf(DP_INFO, "img1_dims:");
	debug_print_dims(DP_INFO, DIMS, img1_dims);
	

	// Data will be X Y 1 Channles 1 Echos
	unsigned int HSquashFlags=32;
	unsigned int DeHSquashFlags=64;

	long HMergedDims[DIMS];
	md_merge_dims(DIMS, HMergedDims, img1_dims, Hankelization_dims);
	
	long HNewDims[DIMS];
	md_select_dims(DIMS, ~HSquashFlags, HNewDims, HMergedDims);
	long DeHNewDims[DIMS];
	md_select_dims(DIMS, ~DeHSquashFlags, DeHNewDims, HMergedDims);

	debug_printf(DP_INFO, "HNewDims:");
	debug_print_dims(DP_INFO, DIMS, HNewDims);
	debug_printf(DP_INFO, "DeHNewDims:");
	debug_print_dims(DP_INFO, DIMS, DeHNewDims);
		
	long CurFlags=md_nontriv_dims(DIMS,img1_dims);
	long HNewFlags=md_nontriv_dims(DIMS,HNewDims);
	long DeHNewFlags=md_nontriv_dims(DIMS,DeHNewDims);
	long TFlags=md_nontriv_dims(DIMS,Hankelization_dims);

	const struct linop_s* HankelLinop = linop_fmac_create(DIMS, HMergedDims, ~HNewFlags, ~CurFlags, ~TFlags, HankelizationMat);

	// const struct linop_s* DeHankelLinop = linop_fmac_create(DIMS, HMergedDims, ~DeHNewFlags, ~HNewFlags, ~TFlags, DeHankelizationMat);

	const struct linop_s* PartLinop= linop_PartitionDim_create(DIMS, HNewDims, 6, 7, 3);

	HankelLinop = linop_chain(HankelLinop,PartLinop);
	linop_free(PartLinop);
	
	SetLinop(HankelLinop);
	// Hankelization end

	opt_reg_configure(DIMS, img1_dims, &ropts, thresh_ops, trafos, llr_blk, shift_mode, conf.gpu);

	if (conf.bpsense)
		opt_bpursuit_configure(&ropts, thresh_ops, trafos, forward_op, kspace, bpsense_eps);

	int nr_penalties = ropts.r;
	struct reg_s* regs = ropts.regs;

	// choose algorithm

	if (ALGO_DEFAULT == algo)
		algo = italgo_choose(nr_penalties, regs);

	if (conf.bpsense)
		assert((ALGO_ADMM == algo) || (ALGO_PRIDU == algo));


	// choose step size

	if ((ALGO_IST == algo) || (ALGO_FISTA == algo) || (ALGO_PRIDU == algo)) {

		// For non-Cartesian trajectories, the default
		// will usually not work. TODO: The same is true
		// for sensitivities which are not normalized, but
		// we do not detect this case.

		if (-1. == step)
			step = 0.95;
	}

	if ((ALGO_CG == algo) || (ALGO_ADMM == algo))
		if (-1. != step)
			debug_printf(DP_INFO, "Stepsize ignored.\n");

	step /= maxeigen;


	// initialize algorithm

	struct iter it = italgo_config(algo, nr_penalties, regs, maxiter, step, hogwild, fast, admm, scaling, warm_start);

	if (ALGO_CG == algo)
		nr_penalties = 0;

	bool trafos_cond = (   (ALGO_PRIDU == algo)
			    || (ALGO_ADMM == algo)
			    || (   (ALGO_NIHT == algo)
				&& (regs[0].xform == NIHTWAV)));

	// Linop script: changed here - no pattern	
	printf("Preparing operator\n");
	const struct operator_s* op = sense_recon_create(&conf, max1_dims, forward_op,
				// pat1_dims, (conf.bpsense) ? NULL : pattern1,
				pat1_dims, NULL,
				it.italgo, it.iconf, image_start1, nr_penalties, thresh_ops,
				trafos_cond ? trafos : NULL, precond_op);

    /*const struct operator_s* op = lsqr2_create(&lsqr_conf, italgo, iconf, (const float*)init, sense_op, precond_op,
					num_funs, thresh_op, thresh_funs, NULL);

    linop_free(sense_op);*/
        
	long strsx[2][DIMS];
	const long* strs[2] = { strsx[0], strsx[1] };

	md_calc_strides(DIMS, strsx[0], img_dims, CFL_SIZE);
	md_calc_strides(DIMS, strsx[1], ksp_dims, CFL_SIZE);

	for (unsigned int i = 0; i < DIMS; i++) {

		if (MD_IS_SET(loop_flags, i)) {

			strsx[0][i] = 0;
			strsx[1][i] = 0;
		}
	}

	if (0 != loop_flags) {

		op = operator_copy_wrapper(2, strs, op);
		// op = operator_loop(DIMS, loop_dims, op);
		op = operator_loop_parallel(DIMS, loop_dims, op, loop_flags, conf.gpu);
	}

	const struct operator_s* adjoint=NULL;
	adjoint = operator_ref(forward_op->adjoint);
	
    printf("Now applying\n");
	// Linop script: changed to argv[2]
	// complex float* tmpx = create_cfl("/tmp/asdasdB", DIMS, img_dims);
	// operator_apply(adjoint, DIMS, img_dims, tmpx,	DIMS, ksp_dims, kspace);
	// md_clear(DIMS, img_dims, tmpx, CFL_SIZE);
	// md_copy(DIMS, img_dims, tmpx, image_adj ,CFL_SIZE);
	// unmap_cfl(16, img_dims, tmpx);

    // debug_printf(DP_INFO, "img_dims ");
	// debug_print_dims(DP_INFO,DIMS,img_dims);
	operator_apply(op, DIMS, img_dims, image, DIMS, conf.bpsense ? img_dims : ksp_dims, conf.bpsense ? NULL : kspace);

    printf("Now freeing\n");
	operator_free(op);

	opt_reg_free(&ropts, thresh_ops, trafos);

	italgo_config_free(it);

	if (scale_im)
		md_zsmul(DIMS, img_dims, image, image, scaling);

	// clean up

    // Linop script: removed
// 	unmap_cfl(DIMS, map_dims, maps);
    
	unmap_cfl(DIMS, ksp_dims, kspace);
	unmap_cfl(DIMS, img_dims, image);

	if (im_truth)
		unmap_cfl(DIMS, img_dims, image_truth);

	if (image_start)
		unmap_cfl(DIMS, img_dims, image_start);

	double end_time = timestamp();

	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);

    // Linop script
    ClearReadScriptFiles(&argv[3],num_args-2);
    // end Linop script
    
	return 0;
}


