function res = fft2cg(x)

% res = fft2c(x)
% 
% orthonormal forward 2D FFT
%
% (c) Michael Lustig 2005

% res = 1/sqrt(prod(gsize(x,1:2)))*fftshift(fft2(ifftshift(x)));
res = 1/sqrt(prod(gsize(x,1:2)))*fftshift(fftshift(fft2(ifftshift(ifftshift(x,1),2)),1),2);

