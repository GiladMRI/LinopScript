function Out=getKrandomSamples(In,K)
if(numel(In)==1)
    In=1:In;
end
P=randperm(numel(In));
Out=In(P(1:K));