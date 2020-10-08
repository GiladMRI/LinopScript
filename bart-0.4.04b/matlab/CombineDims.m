function Out=CombineDims(In,Dims)
% if(size(In,Dims(1))==1)
%     Out=In;
%     return;
% end
if(ndims(In)==2)
    if(Dims(1)==1)
        Out=reshape(In,1,[]);
    else
        if(Dims(1)>2)
            Out=In;
        else
            Out=reshape(In,1,[]);
        end
    end
    return;
end
MinDim=min(Dims);
if(numel(Dims)==2 && Dims(1)~=MinDim)
    Out=CombineDims(permute(In,[1:(MinDim-1) Dims(1) (MinDim+1):(Dims(1)-1) MinDim (Dims(1)+1):ndims(In)]),sort(Dims));
    return;
end
AllDims=1:ndims(In);
OtherDims=setdiff(AllDims,Dims);

Out=permute(In,[OtherDims Dims(end:-1:1)]);
Out=reshape(Out,[gsize(In,OtherDims) prod(gsize(In,Dims)) 1]);
Out=permute(Out,[1:(MinDim-1) AllDims(end)-1 Dims(1):(AllDims(end)-2)]);
