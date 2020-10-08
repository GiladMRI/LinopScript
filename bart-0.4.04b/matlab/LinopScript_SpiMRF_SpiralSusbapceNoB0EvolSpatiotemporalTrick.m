addpath('/autofs/space/daisy_002/users/Gilad/bart-0.4.04b/matlab/');

setenv('TOOLBOX_PATH','/autofs/space/daisy_002/users/Gilad/bart-0.4.04b/');

BaseSP='/autofs/space/daisy_002/users/Gilad';

setenv('TEMP_PATH','/dev/shm/');

ToBARTP=['/dev/shm/LS_' num2str(floor(rand*1e10)) '_'];

SampMaskFN=[ToBARTP 'Msk'];
SensFN=[ToBARTP 'Sens'];
CompsFN=[ToBARTP 'Comps'];
B0FN=[ToBARTP 'B0'];
SigFN=[ToBARTP 'Sig'];
TSBFN=[ToBARTP 'TSB'];
LPAFN=[ToBARTP 'LPA'];
KernFN=[ToBARTP 'Kern'];

AHSigFN=[ToBARTP 'AHSig'];
%% load senstivity maps and quantitative maps (PD phase, angle, B0, R2, and R2*)
load('ExampleSensAnsMaps.mat');
%% Simulate data
% We use echoes and T2* although for MR physics it doesn't make sense,
% Better to think of it as different TIs, etc.
% All different images ("echoes") have the same background phase, but
% different contrasts.
nEchoes=40;
nExcitations=nEchoes; % to be clear I'll refer to those as "excitations"

N=200;
Sz=[N N];

EchoSpacing_ms=1;

TE0_ms=8;
TEs_ms=TE0_ms+(0:nEchoes-1)*EchoSpacing_ms;

SliceIdx=10;

PD=-MapsGESES(:,:,1,SliceIdx).*exp(1i*MapsGESES(:,:,2,SliceIdx));

PD=imresize(PD,Sz);

T2star_ms=min(200,1./abs(imresize(MapsGESES(:,:,5,SliceIdx),Sz)));

% T2star_ms=T2star_ms*0+1000;

Sens=imresize(SensPCS(:,:,SliceIdx,:),Sz);

DecayDueToT2star=exp(-perm32(TEs_ms)./T2star_ms);
I=PD.*DecayDueToT2star;

IWithSens=I.*Sens;
nPE=size(I,1);
nRO=size(I,2);
% IWithSens [PE, RO, Echoes, Channels]

writecfl(SensFN,Sens);
disp('Preparation done');
%% Subspace components
I2d=reshape(abs(I),[],nEchoes);
[~,SComps,AllComps]=svd(I2d,'econ');

nCompsToUse=3;
Comps=AllComps(:,1:nCompsToUse);
% figure;plot(Comps)
clear I2d

CompsP=repmat(permute(Comps,[3 4 5 6 7 2 1]),[nPE nRO]);
% CompsP: [X Y 1 1 1 Comps Echoes]
% repmat for better GPU performance in BART (fixed?)
writecfl(CompsFN,CompsP);
%% Trajectory
TotalAcqTime_ms=15;
AcqDurPerPoint_us=2;
nPoints=TotalAcqTime_ms*1e3/AcqDurPerPoint_us;

R=9; % Each excitation highly undersampled
nLoops=N/2/R;
TrajBase=linspace(0,N/2,nPoints).*exp(1i*2*pi*linspace(0,nLoops,nPoints));

GoldenAngleDeg=137.5;
TrajAllExcs=TrajBase.*exp(1i*perm32(0:nExcitations-1)*GoldenAngleDeg*pi/180);

figure;plot(squeeze(TrajAllExcs(:,:,1:5)));axis equal
%% get signal
ComplexTo3Rows=@(x) [real(x); imag(x); real(x)*0];
Traj3=ComplexTo3Rows(TrajAllExcs);

SigAllEchoes=bartx('nufft ',perm73(Traj3),perm73(IWithSens));
disp('Simulated the signal');
%% get Toeplitz kernel (for each excitation)
% We use [X Y 1 Channels 1 Comps Excitations LPA]
% LPA is just Linear phase, part of the Toeplitz kernel in BART

Script_NUFFT_ForKernel=[BaseSP 'ForKernel'];
Ops={['nufft 0 -1 -1 7 1 0 0 0 ' ToBARTP ]}; % Will save the kernel in ToBARTP
WriteLinopToFile(Script_NUFFT_ForKernel,Ops);

writecfl(SigFN,zeros(1,nPoints)); % Sig is not important for Kernel calculation

delete([lower(ToBARTP) 'LPA*.*']);
delete([lower(ToBARTP) 'PSF*.*']);
clear LPA PSF
for i=1:nEchoes
    disp(i);
    tmp=bartx(['linopScript -d 0 -A ' Script_NUFFT_ForKernel],FillOnesTo16(Sz),SigFN,Traj3(:,:,i));
    PSF(:,:,:,i)=perm74(squeeze(readcfl([lower(ToBARTP) 'PSF'])));
end
LPA=perm74(squeeze(readcfl([lower(ToBARTP) 'LPA'])));

writecfl(LPAFN,perm83(LPA));

fgmontagex(log(abs(PSF(:,:,1,:)))) % Makes some sense
%% get A^H (signal). Define the image-to-signal operator using NUFFT
% Define the image-to-signal operator
Ops={   '_Comps fmac 0 32',... % Open components: Multiply and sum over the components dimension, e.g. Dim 6, BART flag 32
        '_Sens fmac 1 0',... % Multiply by sensitivity maps
        'nufft 2 -1 -1 7 0 0 0 0'}; % Apply NUFFT
Script_NUFFT_Comps=[BaseSP 'NUFFT_Comps'];
WriteLinopToFile(Script_NUFFT_Comps,Ops);

writecfl(SigFN,SigAllEchoes);

SzComps=[Sz 1 1 1 nCompsToUse];
AHSig=bartx(['linopScript -A -d 0 ' Script_NUFFT_Comps],FillOnesTo16(SzComps),SigFN,CompsFN,SensFN,perm73(Traj3));
writecfl(AHSigFN,AHSig);

ShowAbsAngle(AHSig) % Still makes sense
%% Now spatiotemporal trick
Kern=perm83(sum(PSF.*permute(Comps,[3 4 5 1 6 2]).*permute(conj(Comps),[3 4 5 1 6 7 2]),4));
% Kern: X Y 1 (1: channels) 1 Comps CompsAux LPA
writecfl(KernFN,Kern);
%% Now, reconstruct, given B0, using Toeplitz and spatiotemporal trick
% Entire calculation done in the subspace domain
Ops={   'ident','normal',... % we define the normal operator
        '_Sens fmac 0 0',... % Multiply by sensitivity maps
        '_LPA fmac 1 0',... % Linear phase, part of the Toeplitz kernel in BART
        '_FT fft 3',... % Fourier transform over 2D
        'fmac 2 32',... % Apply the Toeplitz kernel, sum over Comps
        'transpose 5 6',... % Transpose to get back to the required dimentionality
        'a FT','a LPA','a Sens'};
Script_Toep=[BaseSP 'Toep'];
WriteLinopToFile(Script_Toep,Ops);
ArgsForLinop={SensFN,LPAFN,KernFN};

% Use Locally-low-rank over components.
Rec=bartx(['picsS -w 1 -S -g -m -R L:3:3:.01 -d 5  ' Script_Toep],FillOnesTo16(SzComps),AHSigFN,ArgsForLinop{:});

RecEchoes=sum(Rec.*CompsP,6);

ShowAbsAngle(flipud(RecEchoes(:,:,3:5:end))); % Nice!
% We see the T2* decay