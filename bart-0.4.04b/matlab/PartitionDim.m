function P=PartitionDim(In,dim,k)
Sz=size(In);
OtherDims=setdiff(1:ndims(In),dim);
P=permute(In, [dim, OtherDims]);
P=reshape(P,[Sz(dim)/k k Sz(OtherDims)]);
P=permute(P,[1 3:(ndims(In)+1) 2]);
P=ipermute(P,[dim, OtherDims ndims(In)+1]);
