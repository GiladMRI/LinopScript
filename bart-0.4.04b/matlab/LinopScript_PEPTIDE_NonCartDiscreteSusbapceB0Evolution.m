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
N=200;
Sz=[N N];

EchoSpacing_ms=1;

nEchoes=40;
TE0_ms=8;
TEs_ms=TE0_ms+(0:nEchoes-1)*EchoSpacing_ms;

SliceIdx=10;

PD=-MapsGESES(:,:,1,SliceIdx).*exp(1i*MapsGESES(:,:,2,SliceIdx));

PD=imresize(PD,Sz);
B0_Hz=imresize(MapsGESES(:,:,3,SliceIdx),Sz);

% B0_Hz=B0_Hz*0;

T2star_ms=min(200,1./abs(imresize(MapsGESES(:,:,5,SliceIdx),Sz)));

% T2star_ms=T2star_ms*0+1000;

Sens=imresize(SensPCS(:,:,SliceIdx,:),Sz);

PhaseDueToB0=exp(1i*2*pi*perm32(TEs_ms).*B0_Hz /1000);
DecayDueToT2star=exp(-perm32(TEs_ms)./T2star_ms);
I=PD.*PhaseDueToB0.*DecayDueToT2star;

IWithSens=I.*Sens;
nPE=size(I,1);
nRO=size(I,2);
% IWithSens [PE, RO, Echoes, Channels]

writecfl(SensFN,Sens);
disp('Preparation done');
%% Subspace components
I2d=reshape(abs(I),[],nEchoes);
[~,~,AllComps]=svd(I2d,'econ');

nCompsToUse=3;
Comps=AllComps(:,1:nCompsToUse);
% figure;plot(Comps)
clear I2d

CompsP=repmat(permute(Comps,[3 4 5 6 7 2 1]),[nPE nRO]);
% CompsP: [X Y 1 1 1 Comps Echoes]
% repmat for better GPU performance in BART (fixed?)
writecfl(CompsFN,CompsP);
%% Trajectory
R=4;
BladeWidth=36;
Idx=-BladeWidth/2+mod((1:nEchoes)*R,BladeWidth);

TrajBaseX=repmat(-N/2:N/2-1,[1 1 nEchoes]);
TrajBaseY=repmat(perm32(Idx),[1 N 1]);
TrajBase=TrajBaseX+1i*TrajBaseY;

nBlades=10;
GoldenAngleDeg=111.25;

TrajPerBlade=TrajBase.* exp(1i*perm42(0:nBlades-1)*GoldenAngleDeg*pi/180);
TrajAllBlades=CombineDims(TrajPerBlade,[4 2]);

figure;
subplot(2,2,1);plot(TrajAllBlades(:,:,2),'.');
subplot(2,2,2);plot(TrajAllBlades(:,:,12),'.');
subplot(2,2,3);plot(TrajAllBlades(:,:,17),'.');
subplot(2,2,4);plot(TrajAllBlades(:),'.');
%% get signal
ComplexTo3Rows=@(x) [real(x); imag(x); real(x)*0];
Traj3=ComplexTo3Rows(TrajAllBlades);

SigAllEchoes=bartx('nufft ',perm73(Traj3),perm73(IWithSens));
disp('Simulated the signal');
%% get Toeplitz kernel (for each echo)
% We use [X Y 1 Channels 1 Comps Echoes LPA]
% LPA is just Linear phase, part of the Toeplitz kernel in BART

Script_NUFFT_ForKernel=[BaseSP 'ForKernel'];
Ops={['nufft 0 -1 -1 7 1 0 0 0 ' ToBARTP ]}; % Will save the kernel in ToBARTP
WriteLinopToFile(Script_NUFFT_ForKernel,Ops);

writecfl(SigFN,SigAllEchoes(:,:,:,1,1)*0); % Sig is not important for Kernel calculation

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
writecfl(KernFN,perm83(perm74(real(PSF))));

fgmontagex(log(abs(PSF(:,:,1,:)))) % Makes some sense
%% get A^H (signal). Define the image-to-signal operator using NUFFT
% Define the image-to-signal operator
Ops={   '_Comps fmac 0 32',... % Open components: Multiply and sum over the components dimension, e.g. Dim 6, BART flag 32
        '_Sens fmac 1 0',... % Multiply by sensitivity maps
        '_B0 fmac 2 0',... % Multiply by B0-induced phase evolution, for the time segments
        'nufft 3 -1 -1 7 0 0 0 0'}; % Apply NUFFT
Script_NUFFT_givenB0=[BaseSP 'NUFFT_givenB0'];
WriteLinopToFile(Script_NUFFT_givenB0,Ops);

writecfl(SigFN,SigAllEchoes);
writecfl(B0FN,perm73(PhaseDueToB0));

SzComps=[Sz 1 1 1 nCompsToUse];
AHSig=bartx(['linopScript -A -d 0 ' Script_NUFFT_givenB0],FillOnesTo16(SzComps),SigFN,CompsFN,SensFN,B0FN,perm73(Traj3));
writecfl(AHSigFN,AHSig);

ShowAbsAngle(AHSig) % Still makes sense
%% Now, reconstruct, given B0, using Toeplitz
Ops={   'ident','normal',... % we define the normal operator
        '_Comps fmac 0 32',... % Open components: Multiply and sum over the components dimension, e.g. Dim 6, BART flag 32
        '_Sens fmac 1 0',... % Multiply by sensitivity maps
        '_B0 fmac 2 0',... % Multiply by B0-induced phase evolution, for the time segments
        '_LPA fmac 3 0',... % Linear phase, part of the Toeplitz kernel in BART
        '_FT fft 3',... % Fourier transform over 2D
        'fmac 4 0',... % Apply the Toeplitz kernel
        'a FT','a LPA','a B0','a Sens','a Comps'};
Script_Toep=[BaseSP 'Toep'];
WriteLinopToFile(Script_Toep,Ops);
ArgsForLinop={CompsFN,SensFN,B0FN,LPAFN,KernFN};

% Use Locally-low-rank over components
Rec=bartx(['picsS -w 1 -S -g -m -R L:3:3:.01 -d 5  ' Script_Toep],FillOnesTo16(SzComps),AHSigFN,ArgsForLinop{:});

RecEchoes=sum(Rec.*CompsP,6).*perm73(PhaseDueToB0);

ShowAbsAngle(flipud(RecEchoes(:,:,3:5:end)));
% We see the T2* decay,
% B0 Helped overcome artifacts...