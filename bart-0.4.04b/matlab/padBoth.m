function Out=padBoth(In,N,dim)
Out=padRight(padLeft(In,N,dim),N,dim);