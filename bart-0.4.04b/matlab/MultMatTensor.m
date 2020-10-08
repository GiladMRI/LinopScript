function Out=MultMatTensor(M,T)
% Sz=size(T);
% Out=zeros([size(M,1) gsize(T,2:numel(Sz))]);
Out=zeros([size(M,1) gsize(T,2:ndims(T))]);
Out(:)=M*reshape(T,size(T,1),[]);
% for i=1:size(T,3)
%     for j=1:size(T,4)
%         for k=1:size(T,5)
%             for l=1:size(T,6)
%                 Out(:,:,i,j,k,l)=M*T(:,:,i,j,k,l);
%             end
%         end
%     end
% end