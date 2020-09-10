function Out=gflip(In,Dims)
Out=In;
for i=1:numel(Dims)
    Out=flip(Out,Dims(i));
end