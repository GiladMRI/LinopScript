function Out=PartitionDimInterleaved(In,Dim,K)
OtherDims=setdiff(1:ndims(In),Dim);
Sz=size(In);
P=permute(In, [Dim ndims(In)+1 OtherDims]);
B=zeros([Sz(Dim)/K K Sz(OtherDims)]);
for i=1:K
    B(:,i,:,:,:,:,:,:,:,:,:)=P(i:K:end,:,:,:,:,:,:,:,:,:);
end
Out=permute(B,[3:Dim-1+2 1 Dim+2:ndims(B) 2]);