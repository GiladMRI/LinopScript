function Out=getPercentile(In,p,n)
if(nargin<3)
    n=numel(In);
end
All=In(isfinite(In));
n=min(n,numel(All));
S=sort(getKrandomSamples(All,n));
Out=S(floor(n*p));