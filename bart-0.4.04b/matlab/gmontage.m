function h=gmontage(In,varargin)
if(~isreal(In))
    warning('Showing abs');
    In=abs(In);
end
if(ndims(In)>4)
    In=squeeze(In);
end
if(prod(gsize(In,3:ndims(In)))==1)
    if(nargin>1)
        imagesc(In,varargin{1});colormap gray;removeTicks;
    else
        imagesc(In);colormap gray;removeTicks;
    end
    return;
end
if(size(In,4)==1)
    In=permute(In,[1 2 4 3]);
else
    if(size(In,3)>1)
        Sz=size(In);
        In=reshape(In,[Sz(1:2) 1 Sz(3)*Sz(4)]);
        varargin{end+1}='Size';
        varargin{end+1}=Sz([4 3]);
    end
end
% In=In./max(In(:));
if(mod(numel(varargin),2)==0)
    varargin=[ggetRange(In) varargin];
end
narginx=numel(varargin)+1;
if(narginx>1)
    In=max(In,varargin{1}(1));
    In=min(In,varargin{1}(2));
    h=montage(In,'DisplayRange',varargin{:});
else
    h=montage(In);
end