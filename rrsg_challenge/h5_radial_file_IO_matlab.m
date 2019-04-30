%% File I/O demo for radial multicoil brain data
% March 2019
% By: Martin Uecker (martin.uecker@med.uni-goettingen.de) and Florian Knoll (florian.knoll@nyumc.org)
%
% Demo data for reproducible research study group initiative to reproduce [1]
%
% [1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
% Advances in sensitivity encoding with arbitrary k-space trajectories.
% Magn Reson Med 46: 638-651 (2001)

clear all; clc; close all; 

% Path to BART toolbox
addpath(genpath('/path/to/bart/'));
setenv('TOOLBOX_PATH', '/path/to/bart/');

%% Load data
rawdata_real    = h5read('rawdata_brain_radial_96proj_12ch.h5','/rawdata');
trajectory      = h5read('rawdata_brain_radial_96proj_12ch.h5','/trajectory');
% rawdata_real    = h5read('rawdata_heart_radial_55proj_34ch.h5','/rawdata');
% trajectory      = h5read('rawdata_heart_radial_55proj_34ch.h5','/trajectory');

rawdata = rawdata_real.r+1i*rawdata_real.i; clear rawdata_real;
rawdata = permute(rawdata,[4,3,2,1]); % Dimension convention of BART
trajectory = permute(trajectory,[3,2,1]); % Dimension convention of BART
[~,nFE,nSpokes,nCh] = size(rawdata);

%% Display rawdata and trajectory
figure; imshow(brighten(log(1+abs(squeeze(rawdata(1,:,:,1)))),0.8),[]); title('rawdata coil 1');

%% Subsampling
% R = 2;
% trajectory = trajectory(:,:,1:R:nSpokes);
% rawdata = rawdata(:,:,1:R:nSpokes,:);
% [~,nSpokes,~]=size(rawdata);

%% Demo: NUFFT reconstruction with BART
% inverse gridding
img_igrid = bart('nufft -i -t', trajectory, rawdata);

% channel combination
img_igrid_sos = bart('rss 8', img_igrid);

%% Display results
figure; imshow(fliplr(flipud(img_igrid_sos)),[]); title('Regridding SOS reconstruction');




