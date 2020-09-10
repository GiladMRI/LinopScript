function Out=MultTensorMat1(T,M)
Out=conj(perm21(MultMatTensor(M',conj(perm21(T)))));