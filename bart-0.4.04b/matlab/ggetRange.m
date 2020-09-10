function Out=ggetRange(In)
% Out=double([min(In(:)) max(In(:))]);
tmp=In(isfinite(In));
tmp=tmp(tmp~=0);
if(isempty(tmp))
    Out2=1;
else
    Out2=getPercentile(In(In~=0),0.92)*1.7;
end
Out=double([min(In(:)) Out2]);