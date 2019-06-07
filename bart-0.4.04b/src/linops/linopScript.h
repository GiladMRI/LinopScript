/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#ifndef __LINOPSCRIPT_H
#define __LINOPSCRIPT_H

#include "misc/cppwrap.h"
#include "misc/types.h"

#include "linops/linop.h"

extern TYPEID linop_data_s;

typedef void (*lop_fun_t)(const linop_data_t* _data, complex float* dst, const complex float* src);
typedef void (*lop_p_fun_t)(const linop_data_t* _data, float lambda, complex float* dst, const complex float* src);
typedef void (*del_fun_t)(const linop_data_t* _data);

struct operator_s;
struct operator_p_s;

struct linop_s;

long * getFdims(long i);
complex float* getDataFile(long i);

const struct linop_s* getLinopScriptFromFile(const char *FN, long CurDims[]);

#include "misc/cppwrap.h"

#endif // __LINOP_H

