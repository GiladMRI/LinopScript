/* Copyright 2014-2017. The Regents of the University of California.
 * Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __OPTREG_H
#define __OPTREG_H

#include "misc/cppwrap.h"


#define NUM_REGS 10

struct operator_p_s;
struct linop_s;


struct reg_s {

	enum { L1WAV, NIHTWAV, NIHTIM, TV, LLR, MLR, IMAGL1, IMAGL2, L1IMG, L2IMG, FTL1, LAPLACE, POS, HLLR, L2CH, L1LS, L2LS, LRLS, L2FD } xform;

	unsigned int xflags;
	unsigned int jflags;

	float lambda;
	unsigned int k;

	// ggg
	float alpha;
	unsigned int q;
	long k2;
	unsigned int k3;
};


struct opt_reg_s {

	float lambda;
	struct reg_s regs[NUM_REGS];
	unsigned int r;
	unsigned int k;
};



extern _Bool opt_reg_init(struct opt_reg_s* ropts);

extern void opt_bpursuit_configure(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], const struct linop_s* model_op, const _Complex float* data, const float eps);

extern void opt_reg_configure(unsigned int N, const long img_dims[__VLA(N)], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS],
 const struct linop_s* trafos[NUM_REGS], unsigned int llr_blk, unsigned int shift_mode, _Bool use_gpu);

extern void opt_reg_free(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS]);

extern _Bool opt_reg(void* ptr, char c, const char* optarg);

extern void help_reg(void);

void SetLinop(const struct linop_s* L);

#include "misc/cppwrap.h"
#endif
