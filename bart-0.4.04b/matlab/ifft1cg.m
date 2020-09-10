function res = ifft1cg(x,dim)

res = sqrt(size(x,dim))*fftshift(ifft(ifftshift(x, dim), [], dim), dim) ;
% 
% S = size(x);
% fctr = S(1)*S(2);
% 
% x = reshape(x,S(1),S(2),prod(S(3:end)));
% 
% res = zeros(size(x));
% for n=1:size(x,3)
% res(:,:,n) = sqrt(fctr)*fftshift(ifft2(ifftshift(x(:,:,n))));
% end
% 
% 
% res = reshape(res,S);

