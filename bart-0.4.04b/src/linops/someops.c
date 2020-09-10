/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */


#include <complex.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/wavelet.h"
#include "num/conv.h"
#include "num/ops.h"
#include "num/iovec.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"

#include "someops.h"

static DEF_TYPEID(cdiag_s);

struct cdiag_s {

	INTERFACE(linop_data_t);

	unsigned int N;
	const long* dims;
	const long* strs;
	const long* ddims;
	const long* dstrs;
	const complex float* diag;
#ifdef USE_CUDA
	const complex float* gpu_diag;
#endif
	bool rmul;
};


static void cdiag_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(cdiag_s, _data);

	const complex float* diag = data->diag;
#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		if (NULL == data->gpu_diag)
			((struct cdiag_s*)data)->gpu_diag = md_gpu_move(data->N, data->ddims, data->diag, CFL_SIZE);

		diag = data->gpu_diag;
	}
#endif
	(data->rmul ? md_zrmul2 : md_zmul2)(data->N, data->dims, data->strs, dst, data->strs, src, data->dstrs, diag);
}

static void cdiag_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(cdiag_s, _data);

	const complex float* diag = data->diag;
#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		if (NULL == data->gpu_diag)
			((struct cdiag_s*)data)->gpu_diag = md_gpu_move(data->N, data->ddims, data->diag, CFL_SIZE);

		diag = data->gpu_diag;
	}
#endif
	(data->rmul ? md_zrmul2 : md_zmulc2)(data->N, data->dims, data->strs, dst, data->strs, src, data->dstrs, diag);
}

static void cdiag_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	cdiag_apply(_data, dst, src);
	cdiag_adjoint(_data, dst, dst);
}

static void cdiag_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(cdiag_s, _data);

	md_free(data->diag);
#ifdef USE_CUDA
	md_free(data->gpu_diag);
#endif
	xfree(data->ddims);
	xfree(data->dims);
	xfree(data->dstrs);
	xfree(data->strs);

	xfree(data);
}

static struct linop_s* linop_gdiag_create(unsigned int N, const long dims[N], unsigned int flags, const complex float* diag, bool rdiag)
{
	PTR_ALLOC(struct cdiag_s, data);
	SET_TYPEID(cdiag_s, data);

	data->rmul = rdiag;

	data->N = N;
	PTR_ALLOC(long[N], ddims);
	PTR_ALLOC(long[N], dstrs);
	PTR_ALLOC(long[N], dims2);
	PTR_ALLOC(long[N], strs);

	md_select_dims(N, flags, *ddims, dims);
	md_calc_strides(N, *dstrs, *ddims, CFL_SIZE);

	md_copy_dims(N, *dims2, dims);
	md_calc_strides(N, *strs, dims, CFL_SIZE);

	data->dims = *PTR_PASS(dims2);
	data->strs = *PTR_PASS(strs);
	data->ddims = *PTR_PASS(ddims);
	data->dstrs = *PTR_PASS(dstrs);

	complex float* tmp = md_alloc(N, data->ddims, CFL_SIZE);
	md_copy(N, data->ddims, tmp, diag, CFL_SIZE);
	data->diag = tmp;

#ifdef USE_CUDA
	data->gpu_diag = NULL;
#endif

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), cdiag_apply, cdiag_adjoint, cdiag_normal, NULL, cdiag_free);
}



/**
 * Create a operator y = D x where D is a diagonal matrix
 *
 * @param N number of dimensions
 * @param dims input and output dimensions
 * @param flags bitmask specifiying the dimensions present in diag
 * @param diag diagonal matrix
 */
struct linop_s* linop_cdiag_create(unsigned int N, const long dims[N], unsigned int flags, const complex float* diag)
{
	return linop_gdiag_create(N, dims, flags, diag, false);
}


/**
 * Create a operator y = D x where D is a diagonal matrix
 *
 * @param N number of dimensions
 * @param dims input and output dimensions
 * @param flags bitmask specifiying the dimensions present in diag
 * @param diag diagonal matrix
 */
struct linop_s* linop_rdiag_create(unsigned int N, const long dims[N], unsigned int flags, const complex float* diag)
{
	return linop_gdiag_create(N, dims, flags, diag, true);
}



struct identity_data_s {

	INTERFACE(linop_data_t);

	const struct iovec_s* domain;
};

static DEF_TYPEID(identity_data_s);

static void identity_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(identity_data_s, _data);
	const struct iovec_s* domain = data->domain;

	md_copy2(domain->N, domain->dims, domain->strs, dst, domain->strs, src, CFL_SIZE);
}

static void identity_free(const linop_data_t* _data)
{	
	const auto data = CAST_DOWN(identity_data_s, _data);

	iovec_free(data->domain);

	xfree(data);
}

/**
 * Create an Identity linear operator: I x
 * @param N number of dimensions
 * @param dims dimensions of input (domain)
 */
struct linop_s* linop_identity_create(unsigned int N, const long dims[N])
{
	PTR_ALLOC(struct identity_data_s, data);
	SET_TYPEID(identity_data_s, data);

	data->domain = iovec_create(N, dims, CFL_SIZE);

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), identity_apply, identity_apply, identity_apply, NULL, identity_free);
}


struct resize_op_s {

	INTERFACE(linop_data_t);

	unsigned int N;
	const long* out_dims;
	const long* in_dims;
};

static DEF_TYPEID(resize_op_s);

static void resize_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(resize_op_s, _data);

	md_resize_center(data->N, data->out_dims, dst, data->in_dims, src, CFL_SIZE);
}

static void resize_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(resize_op_s, _data);

	md_resize_center(data->N, data->in_dims, dst, data->out_dims, src, CFL_SIZE);
}

static void resize_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(resize_op_s, _data);

	complex float* tmp = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);

	resize_forward(_data, tmp, src);
	resize_adjoint(_data, dst, tmp);

	md_free(tmp);
}

static void resize_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(resize_op_s, _data);

	xfree(data->out_dims);
	xfree(data->in_dims);

	xfree(data);
}


/**
 * Create a resize linear operator: y = M x,
 * where M either crops or expands the the input dimensions to match the output dimensions.
 * Uses centered zero-padding and centered cropping
 *
 * @param N number of dimensions
 * @param out_dims output dimensions
 * @param in_dims input dimensions
 */
struct linop_s* linop_resize_create(unsigned int N, const long out_dims[N], const long in_dims[N])
{
	PTR_ALLOC(struct resize_op_s, data);
	SET_TYPEID(resize_op_s, data);

	data->N = N;
	data->out_dims = *TYPE_ALLOC(long[N]);
	data->in_dims = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, (long*)data->out_dims, out_dims);
	md_copy_dims(N, (long*)data->in_dims, in_dims);

	return linop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), resize_forward, resize_adjoint, resize_normal, NULL, resize_free);
}

static void resizeBase_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(resize_op_s, _data);

	md_resize(data->N, data->out_dims, dst, data->in_dims, src, CFL_SIZE);
}

static void resizeBase_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(resize_op_s, _data);

	md_resize(data->N, data->in_dims, dst, data->out_dims, src, CFL_SIZE);
}

static void resizeBase_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(resize_op_s, _data);

	complex float* tmp = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);

	resizeBase_forward(_data, tmp, src);
	resizeBase_adjoint(_data, dst, tmp);

	md_free(tmp);
}

struct linop_s* linop_resizeBase_create(unsigned int N, const long out_dims[N], const long in_dims[N])
{
	PTR_ALLOC(struct resize_op_s, data);
	SET_TYPEID(resize_op_s, data);

	data->N = N;
	data->out_dims = *TYPE_ALLOC(long[N]);
	data->in_dims = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, (long*)data->out_dims, out_dims);
	md_copy_dims(N, (long*)data->in_dims, in_dims);

	return linop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), resizeBase_forward, resizeBase_adjoint, resizeBase_normal, NULL, resize_free);
}

struct extract_op_s {

	INTERFACE(linop_data_t);

	int N;
	const long* pos;
	const long* in_dims;
	const long* out_dims;
};

static DEF_TYPEID(extract_op_s);

static void extract_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(extract_op_s, _data);

	md_clear(data->N, data->out_dims, dst, CFL_SIZE);
	md_copy_block(data->N, data->pos, data->out_dims, dst, data->in_dims, src, CFL_SIZE);
}

static void extract_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(extract_op_s, _data);

	md_clear(data->N, data->in_dims, dst, CFL_SIZE);
	md_copy_block(data->N, data->pos, data->in_dims, dst, data->out_dims, src, CFL_SIZE);
}

static void extract_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(extract_op_s, _data);

	xfree(data->out_dims);
	xfree(data->in_dims);
	xfree(data->pos);

	xfree(data);
}

extern struct linop_s* linop_extract_create(unsigned int N, const long pos[N], const long out_dims[N], const long in_dims[N])
{
	PTR_ALLOC(struct extract_op_s, data);
	SET_TYPEID(extract_op_s, data);

	data->N = N;
	data->pos = *TYPE_ALLOC(long[N]);
	data->out_dims = *TYPE_ALLOC(long[N]);
	data->in_dims = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, (long*)data->pos, pos);
	md_copy_dims(N, (long*)data->out_dims, out_dims);
	md_copy_dims(N, (long*)data->in_dims, in_dims);

	return linop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), extract_forward, extract_adjoint, NULL, NULL, extract_free);
}





struct reshape_op_s {

	INTERFACE(linop_data_t);

	unsigned int N;
	const long* dims;
};

static DEF_TYPEID(reshape_op_s);

static void reshape_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct reshape_op_s* data = CAST_DOWN(reshape_op_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);
}

static void reshape_free(const linop_data_t* _data)
{
	const struct reshape_op_s* data = CAST_DOWN(reshape_op_s, _data);

	xfree(data->dims);

	xfree(data);
}


struct linop_s* linop_reshape_create(unsigned int A, const long out_dims[A], int B, const long in_dims[B])
{
	PTR_ALLOC(struct reshape_op_s, data);
	SET_TYPEID(reshape_op_s, data);

	assert(md_calc_size(A, out_dims) == md_calc_size(B, in_dims));

	unsigned int N = A;
	data->N = N;
	long* dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, dims, out_dims);
	data->dims = dims;

	return linop_create(A, out_dims, B, in_dims, CAST_UP(PTR_PASS(data)), reshape_forward, reshape_forward, reshape_forward, NULL, reshape_free);
}



struct transpose_op_s {

	INTERFACE(linop_data_t);

	int N;
	int a;
	int b;
	const long* dims;
};

static DEF_TYPEID(transpose_op_s);

static void transpose_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(transpose_op_s, _data);

	long odims[data->N];
	md_copy_dims(data->N, odims, data->dims);
	odims[data->a] = data->dims[data->b];
	odims[data->b] = data->dims[data->a];

	md_transpose(data->N, data->a, data->b, odims, dst, data->dims, src, CFL_SIZE);
}

static void transpose_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(transpose_op_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);
}

static void transpose_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(transpose_op_s, _data);

	xfree(data->dims);

	xfree(data);
}


struct linop_s* linop_transpose_create(int N, int a, int b, const long dims[N])
{
	assert((0 <= a) && (a < N));
	assert((0 <= b) && (b < N));
	assert(a != b);

	PTR_ALLOC(struct transpose_op_s, data);
	SET_TYPEID(transpose_op_s, data);

	data->N = N;
	data->a = a;
	data->b = b;

	long* idims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, idims, dims);
	data->dims = idims;

	long odims[N];
	md_copy_dims(N, odims, dims);
	odims[a] = idims[b];
	odims[b] = idims[a];

	return linop_create(N, odims, N, idims, CAST_UP(PTR_PASS(data)), transpose_forward, transpose_forward, transpose_normal, NULL, transpose_free);
}


struct operator_matrix_s {

	INTERFACE(linop_data_t);

	const complex float* mat;
	const complex float* mat_gram; // A^H A
#ifdef USE_CUDA
	const complex float* mat_gpu;
	const complex float* mat_gram_gpu;
#endif
	unsigned int N;

	const long* mat_dims;
	const long* out_dims;
	const long* in_dims;

	const long* grm_dims;
	const long* gin_dims;
	const long* gout_dims;
};

static DEF_TYPEID(operator_matrix_s);


static void linop_matrix_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(operator_matrix_s, _data);
	const complex float* mat = data->mat;

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		if (NULL == data->mat_gpu)
			data->mat_gpu = md_gpu_move(data->N, data->mat_dims, data->mat, CFL_SIZE);

		mat = data->mat_gpu;
	}
#endif

	md_ztenmul(data->N, data->out_dims, dst, data->in_dims, src, data->mat_dims, mat);
}

static void linop_matrix_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(operator_matrix_s, _data);
	const complex float* mat = data->mat;

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		if (NULL == data->mat_gpu)
			data->mat_gpu = md_gpu_move(data->N, data->mat_dims, data->mat, CFL_SIZE);

		mat = data->mat_gpu;
	}
#endif

	md_ztenmulc(data->N, data->in_dims, dst, data->out_dims, src, data->mat_dims, mat);
}

static void linop_matrix_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(operator_matrix_s, _data);

	if (NULL == data->mat_gram) {

		complex float* tmp = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, src);

		linop_matrix_apply(_data, tmp, src);
		linop_matrix_apply_adjoint(_data, dst, tmp);

		md_free(tmp);

	} else {

		const complex float* mat_gram = data->mat_gram;
#ifdef USE_CUDA
		if (cuda_ondevice(src)) {

			if (NULL == data->mat_gram_gpu)
				data->mat_gram_gpu = md_gpu_move(2 * data->N, data->grm_dims, data->mat_gram, CFL_SIZE);

			mat_gram = data->mat_gram_gpu;
		}
#endif
		md_ztenmul(2 * data->N, data->gout_dims, dst, data->gin_dims, src, data->grm_dims, mat_gram);
	}
}

static void linop_matrix_del(const linop_data_t* _data)
{
	auto data = CAST_DOWN(operator_matrix_s, _data);

	xfree(data->out_dims);
	xfree(data->mat_dims);
	xfree(data->in_dims);
	xfree(data->gin_dims);
	xfree(data->gout_dims);
	xfree(data->grm_dims);

	md_free(data->mat);
	md_free(data->mat_gram);
#ifdef USE_CUDA
	md_free(data->mat_gpu);
	md_free(data->mat_gram_gpu);
#endif
	xfree(data);
}


static void shadow_dims(unsigned int N, long out[2 * N], const long in[N])
{
	for (unsigned int i = 0; i < N; i++) {

		out[2 * i + 0] = in[i];
		out[2 * i + 1] = 1;
	}
}


/* O I M G
 * 1 1 1 1   - not used
 * 1 1 A !   - forbidden
 * 1 A 1 !   - forbidden
 * A 1 1 !   - forbidden
 * A A 1 1   - replicated
 * A 1 A 1   - output
 * 1 A A A/A - input
 * A A A A   - batch
 */
static struct operator_matrix_s* linop_matrix_priv2(unsigned int N, const long out_dims[N], const long in_dims[N], const long matrix_dims[N], const complex float* matrix)
{
	// to get assertions and cost estimate

	long max_dims[N];
	md_tenmul_dims(N, max_dims, out_dims, in_dims, matrix_dims);


	PTR_ALLOC(struct operator_matrix_s, data);
	SET_TYPEID(operator_matrix_s, data);

	data->N = N;

	PTR_ALLOC(long[N], out_dims1);
	md_copy_dims(N, *out_dims1, out_dims);
	data->out_dims = *PTR_PASS(out_dims1);

	PTR_ALLOC(long[N], mat_dims1);
	md_copy_dims(N, *mat_dims1, matrix_dims);
	data->mat_dims = *PTR_PASS(mat_dims1);

	PTR_ALLOC(long[N], in_dims1);
	md_copy_dims(N, *in_dims1, in_dims);
	data->in_dims = *PTR_PASS(in_dims1);


	complex float* mat = md_alloc(N, matrix_dims, CFL_SIZE);

	md_copy(N, matrix_dims, mat, matrix, CFL_SIZE);

	data->mat = mat;
	data->mat_gram = NULL;
#ifdef USE_CUDA
	data->mat_gpu = NULL;
	data->mat_gram_gpu = NULL;
#endif

#if 1
	// pre-multiply gram matrix (if there is a cost reduction)

	unsigned long out_flags = md_nontriv_dims(N, out_dims);
	unsigned long in_flags = md_nontriv_dims(N, in_dims);

	unsigned long del_flags = in_flags & ~out_flags;
	unsigned long new_flags = out_flags & ~in_flags;

	/* we double (again) for the gram matrix
	 */

	PTR_ALLOC(long[2 * N], mat_dims2);
	PTR_ALLOC(long[2 * N], in_dims2);
	PTR_ALLOC(long[2 * N], gmt_dims2);
	PTR_ALLOC(long[2 * N], gin_dims2);
	PTR_ALLOC(long[2 * N], grm_dims2);
	PTR_ALLOC(long[2 * N], gout_dims2);

	shadow_dims(N, *gmt_dims2, matrix_dims);
	shadow_dims(N, *mat_dims2, matrix_dims);
	shadow_dims(N, *in_dims2, in_dims);
	shadow_dims(N, *gout_dims2, in_dims);
	shadow_dims(N, *gin_dims2, in_dims);
	shadow_dims(N, *grm_dims2, matrix_dims);

	/* move removed input dims into shadow position
	 * for the gram matrix can have an output there
	 */
	for (unsigned int i = 0; i < N; i++) {

		if (MD_IS_SET(del_flags, i)) {

			assert((*mat_dims2)[2 * i + 0] == (*in_dims2)[2 * i + 0]);

			(*mat_dims2)[2 * i + 1] = (*mat_dims2)[2 * i + 0];
			(*mat_dims2)[2 * i + 0] = 1;

			(*in_dims2)[2 * i + 1] = (*gin_dims2)[2 * i + 0];
			(*in_dims2)[2 * i + 0] = 1;
		}
	}

	for (unsigned int i = 0; i < N; i++) {

		if (MD_IS_SET(new_flags, i)) {

			(*grm_dims2)[2 * i + 0] = 1;
			(*grm_dims2)[2 * i + 1] = 1;
		}

		if (MD_IS_SET(del_flags, i)) {

			(*gout_dims2)[2 * i + 1] = (*gin_dims2)[2 * i + 0];
			(*gout_dims2)[2 * i + 0] = 1;

			(*grm_dims2)[2 * i + 0] = in_dims[i];
			(*grm_dims2)[2 * i + 1] = in_dims[i];
		}
	}


	long gmx_dims[2 * N];
	md_tenmul_dims(2 * N, gmx_dims, *gout_dims2, *gin_dims2, *grm_dims2);

	long mult_mat = md_calc_size(N, max_dims);
	long mult_gram = md_calc_size(2 * N, gmx_dims);

	if (mult_gram < 2 * mult_mat) {	// FIXME: rethink

		debug_printf(DP_DEBUG2, "Gram matrix: 2x %ld vs %ld\n", mult_mat, mult_gram);

		complex float* mat_gram = md_alloc(2 * N, *grm_dims2, CFL_SIZE);

		md_ztenmulc(2 * N, *grm_dims2, mat_gram, *gmt_dims2, matrix, *mat_dims2, matrix);

		data->mat_gram = mat_gram;
	}

	PTR_FREE(gmt_dims2);
	PTR_FREE(mat_dims2);
	PTR_FREE(in_dims2);

	data->gin_dims = *PTR_PASS(gin_dims2);
	data->gout_dims = *PTR_PASS(gout_dims2);
	data->grm_dims = *PTR_PASS(grm_dims2);
#else
	data->gin_dims = NULL;
	data->gout_dims = NULL;
	data->grm_dims = NULL;
#endif

	return PTR_PASS(data);
}


static struct operator_matrix_s* linop_matrix_priv(unsigned int N, const long out_dims[N], const long in_dims[N], const long matrix_dims[N], const complex float* matrix)
{
	unsigned long out_flags = md_nontriv_dims(N, out_dims);
	unsigned long in_flags = md_nontriv_dims(N, in_dims);

	unsigned long del_flags = in_flags & ~out_flags;

	/* we double dimensions for chaining which can lead to
	 * matrices with the same input and output dimension
	 */

	long out_dims2[2 * N];
	long mat_dims2[2 * N];
	long in_dims2[2 * N];

	shadow_dims(N, out_dims2, out_dims);
	shadow_dims(N, mat_dims2, matrix_dims);
	shadow_dims(N, in_dims2, in_dims);

	/* move removed input dims into shadow position
	 * which makes chaining easier below
	 */
	for (unsigned int i = 0; i < N; i++) {

		if (MD_IS_SET(del_flags, i)) {

			assert(1 == out_dims2[2 * i + 0]);
			assert(mat_dims2[2 * i + 0] == in_dims2[2 * i + 0]);

			mat_dims2[2 * i + 1] = mat_dims2[2 * i + 0];
			mat_dims2[2 * i + 0] = 1;

			in_dims2[2 * i + 1] = in_dims[i];
			in_dims2[2 * i + 0] = 1;
		}
	}

	return linop_matrix_priv2(2 * N, out_dims2, in_dims2, mat_dims2, matrix);
}



/**
 * Operator interface for a true matrix:
 * out = mat * in
 * in:	[x x x x 1 x x K x x]
 * mat:	[x x x x T x x K x x]
 * out:	[x x x x T x x 1 x x]
 * where the x's are arbitrary dimensions and T and K may be transposed
 *
 * @param N number of dimensions
 * @param out_dims output dimensions after applying the matrix (codomain)
 * @param in_dims input dimensions to apply the matrix (domain)
 * @param matrix_dims dimensions of the matrix
 * @param matrix matrix data
 */
struct linop_s* linop_matrix_create(unsigned int N, const long out_dims[N], const long in_dims[N], const long matrix_dims[N], const complex float* matrix)
{
	struct operator_matrix_s* data = linop_matrix_priv(N, out_dims, in_dims, matrix_dims, matrix);

	return linop_create(N, out_dims, N, in_dims, CAST_UP(data),
			linop_matrix_apply, linop_matrix_apply_adjoint,
			linop_matrix_apply_normal, NULL, linop_matrix_del);
}


/**
 * Efficiently chain two matrix linops by multiplying the actual matrices together.
 * Stores a copy of the new matrix.
 * Returns: C = B A
 *
 * @param a first matrix (applied to input)
 * @param b second matrix (applied to output of first matrix)
 */
struct linop_s* linop_matrix_chain(const struct linop_s* a, const struct linop_s* b)
{
	const auto a_data = CAST_DOWN(operator_matrix_s, linop_get_data(a));
	const auto b_data = CAST_DOWN(operator_matrix_s, linop_get_data(b));

	// check compatibility
	assert(linop_codomain(a)->N == linop_domain(b)->N);
	assert(md_check_compat(linop_codomain(a)->N, 0u, linop_codomain(a)->dims, linop_domain(b)->dims));

	unsigned int D = linop_domain(a)->N;

	unsigned long outB_flags = md_nontriv_dims(D, linop_codomain(b)->dims);
	unsigned long inB_flags = md_nontriv_dims(D, linop_domain(b)->dims);

	unsigned long delB_flags = inB_flags & ~outB_flags;

	unsigned int N = a_data->N;
	assert(N == 2 * D);

	long in_dims[N];
	md_copy_dims(N, in_dims, a_data->in_dims);

	long matA_dims[N];
	md_copy_dims(N, matA_dims, a_data->mat_dims);

	long matB_dims[N];
	md_copy_dims(N, matB_dims, b_data->mat_dims);

	long out_dims[N];
	md_copy_dims(N, out_dims, b_data->out_dims);

	for (unsigned int i = 0; i < D; i++) {

		if (MD_IS_SET(delB_flags, i)) {

			matA_dims[2 * i + 0] = a_data->mat_dims[2 * i + 1];
			matA_dims[2 * i + 1] = a_data->mat_dims[2 * i + 0];

			in_dims[2 * i + 0] = a_data->in_dims[2 * i + 1];
			in_dims[2 * i + 1] = a_data->in_dims[2 * i + 0];
		}
	}


	long matrix_dims[N];
	md_singleton_dims(N, matrix_dims);

	unsigned long iflags = md_nontriv_dims(N, in_dims);
	unsigned long oflags = md_nontriv_dims(N, out_dims);
	unsigned long flags = iflags | oflags;

	// we combine a and b and sum over dims not in input or output

	md_max_dims(N, flags, matrix_dims, matA_dims, matB_dims);

	debug_printf(DP_DEBUG1, "tensor chain: %ld x %ld -> %ld\n",
			md_calc_size(N, matA_dims), md_calc_size(N, matB_dims), md_calc_size(N, matrix_dims));


	complex float* matrix = md_alloc(N, matrix_dims, CFL_SIZE);

	debug_print_dims(DP_DEBUG2, N, matrix_dims);
	debug_print_dims(DP_DEBUG2, N, in_dims);
	debug_print_dims(DP_DEBUG2, N, matA_dims);
	debug_print_dims(DP_DEBUG2, N, matB_dims);
	debug_print_dims(DP_DEBUG2, N, out_dims);

	md_ztenmul(N, matrix_dims, matrix, matA_dims, a_data->mat, matB_dims, b_data->mat);

	// priv2 takes our doubled dimensions

	struct operator_matrix_s* data = linop_matrix_priv2(N, out_dims, in_dims, matrix_dims, matrix);

	/* although we internally use different dimensions we define the
	 * correct interface
	 */
	struct linop_s* c = linop_create(linop_codomain(b)->N, linop_codomain(b)->dims,
			linop_domain(a)->N, linop_domain(a)->dims, CAST_UP(data),
			linop_matrix_apply, linop_matrix_apply_adjoint,
			linop_matrix_apply_normal, NULL, linop_matrix_del);

	md_free(matrix);

	return c;
}





struct fft_linop_s {

	INTERFACE(linop_data_t);

	const struct operator_s* frw;
	const struct operator_s* adj;

	bool center;
	float nscale;
	
	int N;
	long* dims;
	long* strs;
};

static DEF_TYPEID(fft_linop_s);

static void fft_linop_apply(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(fft_linop_s, _data);

	if (in != out)
		md_copy2(data->N, data->dims, data->strs, out, data->strs, in, CFL_SIZE);

	operator_apply(data->frw, data->N, data->dims, out, data->N, data->dims, out);
}

static void fft_linop_adjoint(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(fft_linop_s, _data);

	if (in != out)
		md_copy2(data->N, data->dims, data->strs, out, data->strs, in, CFL_SIZE);

	operator_apply(data->adj, data->N, data->dims, out, data->N, data->dims, out);
}

static void fft_linop_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(fft_linop_s, _data);

	fft_free(data->frw);
	fft_free(data->adj);

	xfree(data->dims);
	xfree(data->strs);

	xfree(data);
}

static void fft_linop_normal(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(fft_linop_s, _data);

	if (data->center)
		md_copy(data->N, data->dims, out, in, CFL_SIZE);
	else
		md_zsmul(data->N, data->dims, out, in, data->nscale);
}


static struct linop_s* linop_fft_create_priv(int N, const long dims[N], unsigned int flags, bool forward, bool center)
{
	const struct operator_s* plan = fft_measure_create(N, dims, flags, true, false);
	const struct operator_s* iplan = fft_measure_create(N, dims, flags, true, true);

	PTR_ALLOC(struct fft_linop_s, data);
	SET_TYPEID(fft_linop_s, data);

	data->frw = plan;
	data->adj = iplan;
	data->N = N;

	data->center = center;

	data->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->dims, dims);

	data->strs = *TYPE_ALLOC(long[N]);
	md_calc_strides(N, data->strs, data->dims, CFL_SIZE);

	long fft_dims[N];
	md_select_dims(N, flags, fft_dims, dims);
	data->nscale = (float)md_calc_size(N, fft_dims);

	lop_fun_t apply = forward ? fft_linop_apply : fft_linop_adjoint;
	lop_fun_t adjoint = forward ? fft_linop_adjoint : fft_linop_apply;

	struct linop_s* lop =  linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), apply, adjoint, fft_linop_normal, NULL, fft_linop_free);

	if (center) {

		// FIXME: should only allocate flagged dims

		complex float* fftmod_mat = md_alloc(N, dims, CFL_SIZE);
		complex float* fftmodk_mat = md_alloc(N, dims, CFL_SIZE);

		// we need fftmodk only because we want to apply scaling only once

		complex float one[1] = { 1. };
		md_fill(N, dims, fftmod_mat, one, CFL_SIZE);
		fftmod(N, dims, flags, fftmodk_mat, fftmod_mat);
		fftscale(N, dims, flags, fftmod_mat, fftmodk_mat);

		struct linop_s* mod = linop_cdiag_create(N, dims, ~0u, fftmod_mat);
		struct linop_s* modk = linop_cdiag_create(N, dims, ~0u, fftmodk_mat);

		struct linop_s* tmp = linop_chain(mod, lop);
		tmp = linop_chain(tmp, modk);

		linop_free(lop);
		linop_free(mod);
		linop_free(modk);

		lop = tmp;
	}

	return lop;
}


/**
 * Uncentered forward Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 * @param gpu use gpu
 */
struct linop_s* linop_fft_create(int N, const long dims[N], unsigned int flags)
{
	return linop_fft_create_priv(N, dims, flags, true, false);
}


/**
 * Uncentered inverse Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_ifft_create(int N, const long dims[N], unsigned int flags)
{
	return linop_fft_create_priv(N, dims, flags, false, false);
}


/**
 * Centered forward Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_fftc_create(int N, const long dims[N], unsigned int flags)
{
	return linop_fft_create_priv(N, dims, flags, true, true);
}


/**
 * Centered inverse Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_ifftc_create(int N, const long dims[N], unsigned int flags)
{
	return linop_fft_create_priv(N, dims, flags, false, true);
}




struct linop_cdf97_s {

	INTERFACE(linop_data_t);

	unsigned int N;
	const long* dims;
	unsigned int flags;
};

static DEF_TYPEID(linop_cdf97_s);

static void linop_cdf97_apply(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(linop_cdf97_s, _data);

	md_copy(data->N, data->dims, out, in, CFL_SIZE);
	md_cdf97z(data->N, data->dims, data->flags, out);
}

static void linop_cdf97_adjoint(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(linop_cdf97_s, _data);

	md_copy(data->N, data->dims, out, in, CFL_SIZE);
	md_icdf97z(data->N, data->dims, data->flags, out);
}

static void linop_cdf97_normal(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const auto data = CAST_DOWN(linop_cdf97_s, _data);

	md_copy(data->N, data->dims, out, in, CFL_SIZE);
}

static void linop_cdf97_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(linop_cdf97_s, _data);

	xfree(data->dims);

	xfree(data);
}



/**
 * Wavelet CFD9/7 transform operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_cdf97_create(int N, const long dims[N], unsigned int flags)
{
	PTR_ALLOC(struct linop_cdf97_s, data);
	SET_TYPEID(linop_cdf97_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *ndims;
	data->flags = flags;

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), linop_cdf97_apply, linop_cdf97_adjoint, linop_cdf97_normal, NULL, linop_cdf97_free);
}



struct conv_data_s {

	INTERFACE(linop_data_t);

	struct conv_plan* plan;
};

static DEF_TYPEID(conv_data_s);

static void linop_conv_forward(const linop_data_t* _data, complex float* out, const complex float* in)
{
	auto data = CAST_DOWN(conv_data_s, _data);

	conv_exec(data->plan, out, in);
}

static void linop_conv_adjoint(const linop_data_t* _data, complex float* out, const complex float* in)
{
	auto data = CAST_DOWN(conv_data_s, _data);

	conv_adjoint(data->plan, out, in);
}

static void linop_conv_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(conv_data_s, _data);

	conv_free(data->plan);

	xfree(data);
}


/**
 * Convolution operator
 *
 * @param N number of dimensions
 * @param flags bitmask of the dimensions to apply convolution
 * @param ctype
 * @param cmode
 * @param odims output dimensions
 * @param idims input dimensions 
 * @param kdims kernel dimensions
 * @param krn convolution kernel 
 */
struct linop_s* linop_conv_create(int N, unsigned int flags, enum conv_type ctype, enum conv_mode cmode, const long odims[N],
                const long idims[N], const long kdims[N], const complex float* krn)
{
	PTR_ALLOC(struct conv_data_s, data);
	SET_TYPEID(conv_data_s, data);

	data->plan = conv_plan(N, flags, ctype, cmode, odims, idims, kdims, krn);

	return linop_create(N, odims, N, idims, CAST_UP(PTR_PASS(data)), linop_conv_forward, linop_conv_adjoint, NULL, NULL, linop_conv_free);
}















// struct transpose_op_s {

// 	INTERFACE(linop_data_t);

// 	unsigned int N;
// 	const long* out_dims;
// 	const long* in_dims;

// 	unsigned int dim1;
// 	unsigned int dim2;
// };

// static DEF_TYPEID(transpose_op_s);

// static void transpose_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
// {
// 	const auto data = CAST_DOWN(transpose_op_s, _data);

// 	md_transpose(data->N, data->dim1, data->dim2, data->out_dims, dst, data->in_dims, src, CFL_SIZE);
// }

// static void transpose_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
// {
// 	const auto data = CAST_DOWN(transpose_op_s, _data);

// 	md_copy(data->N, data->in_dims, dst, src, CFL_SIZE);
// }

// static void transpose_free(const linop_data_t* _data)
// {
// 	const auto data = CAST_DOWN(transpose_op_s, _data);

// 	xfree(data->out_dims);
// 	xfree(data->in_dims);

// 	xfree(data);
// }


// /**
//  * Create a transpose linear operator:
//  *
//  * @param N number of dimensions
//  * @param out_dims output dimensions
//  * @param in_dims input dimensions
//  */
// struct linop_s* linop_transpose_create(unsigned int N, const long in_dims[N], const long dim1,const long dim2)
// {
// 	PTR_ALLOC(struct transpose_op_s, data);
// 	SET_TYPEID(transpose_op_s, data);

// 	long out_dims[N];

// 	data->N = N;
// 	data->out_dims = *TYPE_ALLOC(long[N]);
// 	data->in_dims = *TYPE_ALLOC(long[N]);
// 	data->dim1=dim1;
// 	data->dim2=dim2;
	
// 	md_copy_dims(N, (long*)data->in_dims, in_dims);
// 	md_transpose_dims(N, dim1, dim2, out_dims, in_dims);
// 	md_copy_dims(N, (long*)data->out_dims, out_dims);
	
// 	return linop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), transpose_forward, transpose_forward, transpose_normal, NULL, transpose_free);
// }





struct print_data_s {

	INTERFACE(linop_data_t);

	const struct iovec_s* domain;

	long messageId;
};

static DEF_TYPEID(print_data_s);

static void print_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(print_data_s, _data);
	const struct iovec_s* domain = data->domain;

	debug_printf(DP_INFO, "Linop print Forward %ld : ", data->messageId);
	if(cuda_ondevice(src)) { debug_printf(DP_INFO,"GPU"); } else { debug_printf(DP_INFO,"CPU"); }
	debug_printf(DP_INFO, " -> ");
	if(cuda_ondevice(dst)) { debug_printf(DP_INFO,"GPU"); } else { debug_printf(DP_INFO,"CPU"); }
	debug_printf(DP_INFO, " ");
	
	debug_print_dims(DP_INFO, domain->N, domain->dims);
	
	md_copy2(domain->N, domain->dims, domain->strs, dst, domain->strs, src, CFL_SIZE);
}

static void print_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(print_data_s, _data);
	const struct iovec_s* domain = data->domain;

	debug_printf(DP_INFO, "Linop print Adjoint %ld : ", data->messageId);

	if(cuda_ondevice(src)) { debug_printf(DP_INFO,"GPU"); } else { debug_printf(DP_INFO,"CPU"); }
	debug_printf(DP_INFO, " -> ");
	if(cuda_ondevice(dst)) { debug_printf(DP_INFO,"GPU"); } else { debug_printf(DP_INFO,"CPU"); }
	debug_printf(DP_INFO, " ");

	debug_print_dims(DP_INFO, domain->N, domain->dims);



	md_copy2(domain->N, domain->dims, domain->strs, dst, domain->strs, src, CFL_SIZE);
}

static void print_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(print_data_s, _data);
	const struct iovec_s* domain = data->domain;

	debug_printf(DP_INFO, "Linop print Normal  %ld : ", data->messageId);
	if(cuda_ondevice(src)) { debug_printf(DP_INFO,"GPU"); } else { debug_printf(DP_INFO,"CPU"); }
	debug_printf(DP_INFO, " -> ");
	if(cuda_ondevice(dst)) { debug_printf(DP_INFO,"GPU"); } else { debug_printf(DP_INFO,"CPU"); }
	debug_printf(DP_INFO, " ");

	debug_print_dims(DP_INFO, domain->N, domain->dims);

	md_copy2(domain->N, domain->dims, domain->strs, dst, domain->strs, src, CFL_SIZE);
}

static void print_pinv(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(print_data_s, _data);
	const struct iovec_s* domain = data->domain;

	debug_printf(DP_WARN, "Linop print pinv %ld :", data->messageId);
	debug_print_dims(DP_INFO, domain->N, domain->dims);

	md_copy2(domain->N, domain->dims, domain->strs, dst, domain->strs, src, CFL_SIZE);
}

static void print_free(const linop_data_t* _data)
{	
	const auto data = CAST_DOWN(print_data_s, _data);

	debug_printf(DP_WARN, "Linop print free %ld :", data->messageId);

	iovec_free(data->domain);

	xfree(data);
}

/**
 * Create an Identity linear operator: I x
 * print message id
 * @param N number of dimensions
 * @param dims dimensions of input (domain)
 */
struct linop_s* linop_print_create(unsigned int N, const long dims[N], long messageId)
{
	PTR_ALLOC(struct print_data_s, data);
	SET_TYPEID(print_data_s, data);

	data->domain = iovec_create(N, dims, CFL_SIZE);
	data->messageId=messageId;

	debug_printf(DP_WARN, "Linop print create %ld\n", messageId);

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), print_forward, print_adjoint, print_normal, print_pinv, print_free);
}













struct applyBySlice_op_s {

	INTERFACE(linop_data_t);

	long nSlices;
	long N;
	long whichDim;
	long istr;
	long ostr;
	long nTnsrs;

	const struct linop_s* opPerSlice;

	complex float** Tensors;
	struct linop_s** Opsx;
	long *tstrs;
};

static DEF_TYPEID(applyBySlice_op_s);

#define BYSLICE_DBG_LVL DP_DEBUG4
// #define BYSLICE_DBG_LVL DP_INFO
static void applyBySlice_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(applyBySlice_op_s, _data);

	debug_printf(BYSLICE_DBG_LVL, "Byslice Forward %ld slices\n", data->nSlices);
	
	long t;
	for(long i=0;i<data->nSlices;i++) {
		for(long t=0;t<data->nTnsrs;t++) {
			// debug_printf(BYSLICE_DBG_LVL, "BySlice Frwrd Slc %ld/%ld tnsr %ld/%ld\n", i,data->nSlices,t,data->nTnsrs);
			// debug_printf(DP_DEBUG3, "AAX %ld\n", data->Opsx[t]);
			// debug_printf(DP_DEBUG3, "AAY %ld\n", data->Tensors[t]);
			// complex float* ten=&data->Tensors[t][i*data->tstrs[t]];
			// debug_printf(DP_DEBUG3, "AAYB %ld\n", ten);
			set_fmac_tensor(data->Opsx[t], &data->Tensors[t][i*data->tstrs[t]]);
			// set_fmac_tensor(data->Opsx[t], ten);
			// debug_printf(DP_DEBUG3, "AAZ %ld\n", data->Tensors[t]);
		}
		
		// debug_printf(DP_DEBUG3, "BB %ld\n", i);
		operator_apply(	data->opPerSlice->forward,	data->N, linop_codomain(data->opPerSlice)->dims, &dst[i*data->ostr],
													data->N, linop_domain(data->opPerSlice)->dims, &src[i*data->istr]); }

	debug_printf(BYSLICE_DBG_LVL, "Byslice Forward end %ld slices\n", data->nSlices);
}

static void applyBySlice_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(applyBySlice_op_s, _data);

	debug_printf(BYSLICE_DBG_LVL, "Byslice adjoint %ld slices\n", data->nSlices);

	for(long i=0;i<data->nSlices;i++) {
		for(long t=0;t<data->nTnsrs;t++) {
			set_fmac_tensor(data->Opsx[t], &data->Tensors[t][i*data->tstrs[t]]);		}

		operator_apply(	data->opPerSlice->adjoint,	data->N, linop_domain(data->opPerSlice)->dims, &dst[i*data->istr],
													data->N, linop_codomain(data->opPerSlice)->dims, &src[i*data->ostr]); }

	debug_printf(BYSLICE_DBG_LVL, "Byslice adjoint end %ld slices\n", data->nSlices);
}

static void applyBySlice_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(applyBySlice_op_s, _data);

	debug_printf(BYSLICE_DBG_LVL, "Byslice normal %ld slices\n", data->nSlices);

	for(long i=0;i<data->nSlices;i++) {
		for(long t=0;t<data->nTnsrs;t++) {
			// debug_printf(DP_DEBUG3, "BySlice Nrml Slc %ld/%ld tnsr %ld/%ld\n", i,data->nSlices,t,data->nTnsrs);
			set_fmac_tensor(data->Opsx[t], &data->Tensors[t][i*data->tstrs[t]]);		}

		operator_apply(	data->opPerSlice->normal,	data->N, linop_domain(data->opPerSlice)->dims, &dst[i*data->istr],
													data->N, linop_domain(data->opPerSlice)->dims, &src[i*data->istr]); }

	debug_printf(BYSLICE_DBG_LVL, "Byslice normal end %ld slices\n", data->nSlices);
}

static void applyBySlice_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(applyBySlice_op_s, _data);

	linop_free(data->opPerSlice);

	for(long i=0;i<data->nTnsrs;i++) {
		md_free(data->Tensors[i]);
	}

	if(data->nTnsrs>0) {
		xfree(data->Tensors);
		xfree(data->Opsx);
		xfree(data->tstrs);
	}
	
	xfree(data);
}

struct linop_s* linop_applyBySlice_create(int N, const long dims[N], int WhichDim, const struct linop_s* opPerSlice,
											complex float* dataFilesx[], const long tstrs[], const struct linop_s* Opsx[], long nTnsrs)
{
	PTR_ALLOC(struct applyBySlice_op_s, data);
	SET_TYPEID(applyBySlice_op_s, data);

	debug_printf(DP_DEBUG1,"linop_applyBySlice_create start. #Tensors: %ld\n",nTnsrs);

	long nSlices=dims[WhichDim];

	long strs[N];
    md_calc_strides(N, strs, dims, CFL_SIZE);

    long BaseOpDims[N];
    md_copy_dims(N, BaseOpDims, linop_domain(opPerSlice)->dims);
    long BaseOpSlices=BaseOpDims[WhichDim];

    long odims[N];
    md_copy_dims(N, odims, linop_codomain(opPerSlice)->dims);
    odims[WhichDim]=nSlices;

    long ostrs[N];
    md_calc_strides(N, ostrs, odims, CFL_SIZE);
    
	data->istr = strs[WhichDim]*BaseOpSlices/CFL_SIZE;
	data->ostr = ostrs[WhichDim]*BaseOpSlices/CFL_SIZE;
	data->nSlices = nSlices/BaseOpSlices;
	data->opPerSlice = opPerSlice;
	data->N=N;

	data->nTnsrs=nTnsrs;

	if(nTnsrs>0) {
		PTR_ALLOC(long[nTnsrs], ptstrs);
	
		data->Tensors = xmalloc(sizeof(complex float*) * nTnsrs);
		data->Opsx = xmalloc(sizeof(struct linop_s*) * nTnsrs);
		data->tstrs = *PTR_PASS(ptstrs);
		
		for(long i=0;i<nTnsrs;i++) {
	    	data->tstrs[i]=tstrs[i]*BaseOpSlices;
			data->Opsx[i]=Opsx[i];
			if(cuda_ondevice(dataFilesx[i])) {
				data->Tensors[i]=dataFilesx[i];
			} else {
				debug_printf(DP_ERROR, "dataFilesx[%ld] not on GPU\n", i);
				exit(EXIT_SUCCESS);
	    		return 0;
			}
		}
	}

	debug_printf(DP_DEBUG1,"linop_applyBySlice_create: #Slices %ld Base #slices %ld\n",nSlices,BaseOpSlices);


	return linop_create(N, odims, N, dims, CAST_UP(PTR_PASS(data)), applyBySlice_forward, applyBySlice_adjoint, applyBySlice_normal, NULL, applyBySlice_free);
}
