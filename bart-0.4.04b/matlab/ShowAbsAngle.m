function ShowAbsAngle(In,Dir,varargin)
if(nargin<2)
    Dir=1;
end
figure;
if(Dir==2)
    subplot(2,1,1);
else
    subplot(1,2,1);
end
gmontage(abs(In),varargin{:});
s = inputname(1);
title(s);
if(Dir==2)
    subplot(2,1,2);
else
    subplot(1,2,2);
end
if(nargin>3)
    if(strcmp(varargin{1},'Size'))
        gmontage(angle(In),[-pi pi],varargin{:});
    else
        gmontage(angle(In),[-pi pi]);
    end
else
    gmontage(angle(In),[-pi pi]);
end
if(Dir==2)
    subplot(2,1,1);
else
    subplot(1,2,1);
end
