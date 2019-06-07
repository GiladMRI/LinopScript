curD=pwd;
BaseP=[pwd filesep];
setenv('TOOLBOX_PATH',[curD filesep 'bart-0.4.04b']);
addpath([curD filesep 'bart-0.4.04b' filesep 'matlab' filesep]);
cd('rrsg_challenge');
%% Load data
rawdata_real    = h5read('rawdata_brain_radial_96proj_12ch.h5','/rawdata');
trajectory      = h5read('rawdata_brain_radial_96proj_12ch.h5','/trajectory');

rawdata = rawdata_real.r+1i*rawdata_real.i; clear rawdata_real;
rawdata = permute(rawdata,[4,3,2,1]); % Dimension convention of BART
trajectory = permute(trajectory,[3,2,1]); % Dimension convention of BART
[~,nFE,nSpokes,nCh] = size(rawdata);
disp('Loaded data');
cd(curD);
%% Demo: NUFFT reconstruction with BART
% inverse gridding
img_igrid = bart('nufft -i -t', trajectory, rawdata);

% channel combination
img_igrid_sos = bart('rss 8', img_igrid);

%% Display results
figure; imshow(fliplr(flipud(img_igrid_sos)),[]); title('Regridding SOS reconstruction');
%% get sens maps
FF=bart('fft 7',img_igrid);
calib = bart(['ecalib -r ' num2str(20)], FF);
Sens = bart('slice 4 0', calib);
%% Running the linear operator script
%  The linopScript command runs a script file. It needs the size of the
% input tensor.
%  In this case, the script is just applying the sensitivity maps, followed
% by applying nufft. The script is therefore:
%  -----  nuftScript.txt   -----
%  # fmac with file #1 - the sensitity maps, and no squashing
%  # Then NUFT according to the trajectory in file #0 (no weights file, basis file, etc. NUFFT flags 7).
%  fmac 1 0
%  nufft 0 -1 -1 7 0 0 0 0 

Sz=size(img_igrid_sos);
Sz16=FillOnesTo16(Sz);
ScriptFN=[BaseP 'nuftScript.txt'];

NUFT=bart(['linopScript ' ScriptFN],Sz16,img_igrid_sos,trajectory,Sens);
NUFTA=bart(['linopScript -A ' ScriptFN],Sz16,rawdata,trajectory,Sens);
%% Running picsS
%  picsS is pics modified to run on linop scripts.
Rec=bart(['picsS -m -R Q:0.00001 ' ScriptFN],Sz16,rawdata,trajectory,Sens);
figure; imshow(fliplr(flipud(Rec)),[]); title('Slightly regularized recon');

trajectoryCombined=reshape(trajectory,3,[]);
rawdataCombined=reshape(rawdata,1,[],12);

% SnufftStruct = nufft_init(BART2Fes_NUFT_Idxs(trajectoryCombined(1:2,:),Sz), Sz, [6 6], Sz*2);
% Kern=NUFFT_to_Toep_2blocks(SnufftStruct);
load(['RadialDemo.mat'],'Kern');
%% Defining a normal operator
%  Sometimes the Normal operator can be calculated faster than
% forward+adjoint, as in the case of nufft using Toeplitz embeddingcurD=pwd;
BaseP=[pwd filesep];
setenv('TOOLBOX_PATH',[curD filesep 'bart-0.4.04b']);
addpath([curD filesep 'bart-0.4.04b' filesep 'matlab' filesep]);
cd('rrsg_challenge');
%% Load data
rawdata_real    = h5read('rawdata_brain_radial_96proj_12ch.h5','/rawdata');
trajectory      = h5read('rawdata_brain_radial_96proj_12ch.h5','/trajectory');

rawdata = rawdata_real.r+1i*rawdata_real.i; clear rawdata_real;
rawdata = permute(rawdata,[4,3,2,1]); % Dimension convention of BART
trajectory = permute(trajectory,[3,2,1]); % Dimension convention of BART
[~,nFE,nSpokes,nCh] = size(rawdata);
disp('Loaded data');
cd(curD);
%% Demo: NUFFT reconstruction with BART
% inverse gridding
img_igrid = bart('nufft -i -t', trajectory, rawdata);

% channel combination
img_igrid_sos = bart('rss 8', img_igrid);

%% Display results
figure; imshow(fliplr(flipud(img_igrid_sos)),[]); title('Regridding SOS reconstruction');
%% get sens maps
FF=bart('fft 7',img_igrid);
calib = bart(['ecalib -r ' num2str(20)], FF);
Sens = bart('slice 4 0', calib);
%% Running the linear operator script
%  The linopScript command runs a script file. It needs the size of the
% input tensor.
%  In this case, the script is just applying the sensitivity maps, followed
% by applying nufft. The script is therefore:
%  -----  nuftScript.txt   -----
%  # fmac with file #1 - the sensitity maps, and no squashing
%  # Then NUFT according to the trajectory in file #0 (no weights file, basis file, etc. NUFFT flags 7).
%  fmac 1 0
%  nufft 0 -1 -1 7 0 0 0 0 

Sz=size(img_igrid_sos);
Sz16=FillOnesTo16(Sz);
ScriptFN=[BaseP 'nuftScript.txt'];

NUFT=bart(['linopScript ' ScriptFN],Sz16,img_igrid_sos,trajectory,Sens);
NUFTA=bart(['linopScript -A ' ScriptFN],Sz16,rawdata,trajectory,Sens);
%% Running picsS
%  picsS is pics modified to run on linop scripts.
Rec=bart(['picsS -m -R Q:0.00001 ' ScriptFN],Sz16,rawdata,trajectory,Sens);
figure; imshow(fliplr(flipud(Rec)),[]); title('Slightly regularized recon');

trajectoryCombined=reshape(trajectory,3,[]);
rawdataCombined=reshape(rawdata,1,[],12);

% SnufftStruct = nufft_init(BART2Fes_NUFT_Idxs(trajectoryCombined(1:2,:),Sz), Sz, [6 6], Sz*2);
% Kern=NUFFT_to_Toep_2blocks(SnufftStruct);
load(['RadialDemo.mat'],'Kern');
%% Defining a normal operator
%  Sometimes the Normal operator can be calculated faster than
% forward+adjoint, as in the case of nufft using Toeplitz embedding.
%  This script file defines the normal operator specificaly (the lines below
% the NORMAL title):
%  --- nuftScriptN.txt  -----
%  fmac 1 0 # this is linop #0
%  nufft 0 -1 -1 7 0 0 0 0
%  NORMAL 000
%  f 0  # this is calling the forward of linop #0 defined above
%  dblsz 3 # padding
%  fft 3
%  fmac 2 0 # This is multplication with the Toeplitz-based kernel
%  ifft 3
%  halfsz 3
%  a 0 # this is the adjoint of linop #0 defined above
ScriptFN_T=[BaseP 'nuftScriptN.txt'];
RecT=bart(['picsS -m -R Q:0.00001 ' ScriptFN_T],Sz16,rawdata,trajectory,Sens,Kern);
figure; imshow(fliplr(flipud(RecT)),[]); title('Slightly regularized recon, through Toeplitz');

% In this script file we take more use of the ability to call previously
% defined linops
ScriptFN_T2=[BaseP 'nuftScriptN2.txt'];
RecT2=bart(['picsS -m -R Q:0.00001 ' ScriptFN_T2],Sz16,rawdata,trajectory,Sens,Kern);
figure; imshow(fliplr(flipud(RecT2)),[]); title('Slightly regularized recon, through Toeplitz, cleaner script');
%% Example 2: T2-shuffling, basic recon
%  T2-shuffling involves:
%  A_for = @(a) P_for(T_for(F_for(S_for(a))));
%  That is:
%  Applying sensitivity maps.
%  Fourier transform.
%  Temporal unfolding (components to time-points).
%  Multiplying by the sampling pattern.
%  So, the script file is as follows (t2shuffleScript.txt):
%
%  ----- t2shuffleScript.txt ----------
%  # File 0 is sensitivity maps, 1 is sampling pattern
%  # File 2 is components: Phi
%  # PRINT 52
%  FMAC 0 0 Applying sensitivity maps
%  FFT 3            
%  FMAC 2 64     Applying temporal basis (components)
%  FMAC 1 0        Undersampling mask
% 
%  To run that from MATLAB, let's prepare the stuff:
%  (After running demo_t2shuffling_recon.m)

load('T2ShuflDemo.mat','ksp','sens','Phi','masks');
kspP=permute(ksp,[1 2 5 3 6 4]);                %   260   240       1     8     1    80
SensP=permute(sens,[1 2 4 3]);                  %   260   240       1     8
PhiP=permute(Phi,[3 4 5 6 7 1 2]);              %   1     1         1     1     1    80     4
masksP=permute(masks,[1 2 3 5 6 4]);            %   260   240       1     1     1    80

K=size(Phi,2);
Sz=[size(SensP,1) size(SensP,2) 1 1 1 1 K];
Sz16=FillOnesTo16(Sz);
T2ShflScriptFN=[BaseP 't2shuffleScript.txt'];

SensP=bart('fftmod 3',SensP);
kspP=bart('fftmod 3',kspP);
Rho=3000;
Lambda=10;
RecT2Shf1=bart(['picsS -m -b 10 -u ' num2str(Rho) ' -R L:3:3:' num2str(Lambda) ' ' T2ShflScriptFN],Sz16,kspP,SensP,masksP,PhiP);
RecT2Shf1=squeeze(RecT2Shf1);

figure;subplot(2,1,1);imshow(abs(reshape(RecT2Shf1, size(RecT2Shf1,1), [])), []);title('RecT2Shf1 components');
subplot(2,1,2);imshow(angle(reshape(RecT2Shf1, size(RecT2Shf1,1), [])), []);
%% Example 3: T2-shuffling, temporal trick
%  As mentioned in Tamir et al., the computation might be done completely in
%  the components domains, without ever unfolding the full temporal
% resolution. 
%  In such a case, one would like to describe the normal operator
% directly. It is of the common form
%
%  $$A_1A_2A_3MA_1^HA_2^HA_3^H$$
%
%  where $M=XYZ$ is an operator chain done only once.
%  (Toeplitz embedding for NUFFT takes similar form)
%
%  For the T2-shuffling case, preparing the central kernel trick:

PT=sum(masks.*permute(Phi,[3 4 5 1 2]),6);
TPT1=sum(PT.*permute(Phi',[3 4 5 2 6 1]),4);

TPT1p=permute(TPT1,[1 2 3 5 6 4]);

TPT1pBart=permute(TPT1p,[1 2 3 6 7 8 4 5]);     %   260   240       1     1     1     1     4     4
ET=permute(eye(K),[3:8 1 2]);                   %   1     1         1     1     1     1     4     4
%%
T2ShflScriptFN_Normal=[BaseP 't2shuffleScriptN.txt'];
RecT2Shf2=bart(['picsS -m -b 10 -u ' num2str(Rho) ' -R L:3:3:' num2str(Lambda) ' ' T2ShflScriptFN_Normal],Sz16,kspP,SensP,masksP,PhiP,TPT1pBart,ET);
RecT2Shf2=squeeze(RecT2Shf2);

figure;subplot(2,1,1);imshow(abs(reshape(RecT2Shf2, size(RecT2Shf2,1), [])), []);title('RecT2Shf2 components');
subplot(2,1,2);imshow(angle(reshape(RecT2Shf2, size(RecT2Shf2,1), [])), []);.
%  This script file defines the normal operator specificaly (the lines below
% the NORMAL title):
%  --- nuftScriptN.txt  -----
%  fmac 1 0 # this is linop #0
%  nufft 0 -1 -1 7 0 0 0 0
%  NORMAL 000
%  f 0  # this is calling the forward of linop #0 defined above
%  dblsz 3 # padding
%  fft 3
%  fmac 2 0 # This is multplication with the Toeplitz-based kernel
%  ifft 3
%  halfsz 3
%  a 0 # this is the adjoint of linop #0 defined above
ScriptFN_T=[BaseP 'nuftScriptN.txt'];
RecT=bart(['picsS -m -R Q:0.00001 ' ScriptFN_T],Sz16,rawdata,trajectory,Sens,Kern);
figure; imshow(fliplr(flipud(RecT)),[]); title('Slightly regularized recon, through Toeplitz');

% In this script file we take more use of the ability to call previously
% defined linops
ScriptFN_T2=[BaseP 'nuftScriptN2.txt'];
RecT2=bart(['picsS -m -R Q:0.00001 ' ScriptFN_T2],Sz16,rawdata,trajectory,Sens,Kern);
figure; imshow(fliplr(flipud(RecT2)),[]); title('Slightly regularized recon, through Toeplitz, cleaner script');
%% Example 2: T2-shuffling, basic recon
%  T2-shuffling involves:
%  A_for = @(a) P_for(T_for(F_for(S_for(a))));
%  That is:
%  Applying sensitivity maps.
%  Fourier transform.
%  Temporal unfolding (components to time-points).
%  Multiplying by the sampling pattern.
%  So, the script file is as follows (t2shuffleScript.txt):
%
%  ----- t2shuffleScript.txt ----------
%  # File 0 is sensitivity maps, 1 is sampling pattern
%  # File 2 is components: Phi
%  # PRINT 52
%  FMAC 0 0 Applying sensitivity maps
%  FFT 3            
%  FMAC 2 64     Applying temporal basis (components)
%  FMAC 1 0        Undersampling mask
% 
%  To run that from MATLAB, let's prepare the stuff:
%  (After running demo_t2shuffling_recon.m)

load('T2ShuflDemo.mat','ksp','sens','Phi','masks');
kspP=permute(ksp,[1 2 5 3 6 4]);                %   260   240       1     8     1    80
SensP=permute(sens,[1 2 4 3]);                  %   260   240       1     8
PhiP=permute(Phi,[3 4 5 6 7 1 2]);              %   1     1         1     1     1    80     4
masksP=permute(masks,[1 2 3 5 6 4]);            %   260   240       1     1     1    80

K=size(Phi,2);
Sz=[size(SensP,1) size(SensP,2) 1 1 1 1 K];
Sz16=FillOnesTo16(Sz);
T2ShflScriptFN=[BaseP 't2shuffleScript.txt'];

SensP=bart('fftmod 3',SensP);
kspP=bart('fftmod 3',kspP);
Rho=3000;
Lambda=10;
RecT2Shf1=bart(['picsS -m -b 10 -u ' num2str(Rho) ' -R L:3:3:' num2str(Lambda) ' ' T2ShflScriptFN],Sz16,kspP,SensP,masksP,PhiP);
RecT2Shf1=squeeze(RecT2Shf1);

figure;subplot(2,1,1);imshow(abs(reshape(RecT2Shf1, size(RecT2Shf1,1), [])), []);title('RecT2Shf1 components');
subplot(2,1,2);imshow(angle(reshape(RecT2Shf1, size(RecT2Shf1,1), [])), []);
%% Example 3: T2-shuffling, temporal trick
%  As mentioned in Tamir et al., the computation might be done completely in
%  the components domains, without ever unfolding the full temporal
% resolution. 
%  In such a case, one would like to describe the normal operator
% directly. It is of the common form
%
%  $$A_1A_2A_3MA_1^HA_2^HA_3^H$$
%
%  where $M=XYZ$ is an operator chain done only once.
%  (Toeplitz embedding for NUFFT takes similar form)
%
%  For the T2-shuffling case, preparing the central kernel trick:

PT=sum(masks.*permute(Phi,[3 4 5 1 2]),6);
TPT1=sum(PT.*permute(Phi',[3 4 5 2 6 1]),4);

TPT1p=permute(TPT1,[1 2 3 5 6 4]);

TPT1pBart=permute(TPT1p,[1 2 3 6 7 8 4 5]);     %   260   240       1     1     1     1     4     4
ET=permute(eye(K),[3:8 1 2]);                   %   1     1         1     1     1     1     4     4
%%
T2ShflScriptFN_Normal=[BaseP 't2shuffleScriptN.txt'];
RecT2Shf2=bart(['picsS -m -b 10 -u ' num2str(Rho) ' -R L:3:3:' num2str(Lambda) ' ' T2ShflScriptFN_Normal],Sz16,kspP,SensP,masksP,PhiP,TPT1pBart,ET);
RecT2Shf2=squeeze(RecT2Shf2);

figure;subplot(2,1,1);imshow(abs(reshape(RecT2Shf2, size(RecT2Shf2,1), [])), []);title('RecT2Shf2 components');
subplot(2,1,2);imshow(angle(reshape(RecT2Shf2, size(RecT2Shf2,1), [])), []);
