function fgmontagex(varargin)
figure;
tmp=squeeze(varargin{1});
if(~isreal(tmp))
    tmp=abs(tmp);
end
if(ndims(tmp)==3)
%     if(size(tmp,3)==4)
%         tmp=perm43(PartitionDim(tmp,3,2));
%     end
%     if(size(tmp,3)==6)
%         tmp=perm43(PartitionDim(tmp,3,2));
%     end
    if(size(tmp,3)==8)
        tmp=perm43(PartitionDim(tmp,3,2));
    end
%     if(size(tmp,3)==16)
%         tmp=perm43(PartitionDim(tmp,3,4));
%     end
%     tmp=CombineDims(tmp,[3 2]);
end
if(ndims(tmp)==4)
    tmp=CombineDims(CombineDims(tmp,[4 2]),[3 1]);
end
gmontage(tmp,varargin{2:end});
s = inputname(1);
title(s,'Interpreter','None');
removeTicks;daspect([1 1 1]);set(get(gcf,'Children'),'Position',[0.05 0.1 0.9 0.8]);removeBox;