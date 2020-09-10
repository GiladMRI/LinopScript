function Out=gsize(In,WhichDims)
if(nargin<2)
    WhichDims=1:ndims(In);
end
Out=size(In);
Out=[Out ones(1,max(WhichDims)-ndims(In))];
Out=Out(WhichDims);