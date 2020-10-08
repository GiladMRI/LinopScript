function res = ifft2cg(x)

res = sqrt(prod(gsize(x,1:2)))*fftshift(fftshift(ifft2(ifftshift(ifftshift(x,1),2)),1),2);

%     res(:,:,n) = sqrt(fctr)*fftshift(ifft2(ifftshift(x(:,:,n))));
