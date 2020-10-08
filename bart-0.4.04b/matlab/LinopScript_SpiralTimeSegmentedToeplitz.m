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
%% Trajectory
TimePerPoint_ms=2/1000; % 2us
nPoints=nEchoes*EchoSpacing_ms/TimePerPoint_ms;

R=2;
nLoops=N/2/R;

Traj=linspace(0,N/2,nPoints).*exp(1i*2*pi*linspace(0,nLoops,nPoints));

% Time segments for simulating the signal
TSB_ForSigSim=GetTSCoeffsByLinear(nPoints,nEchoes);
TSB_ForSigSim=permute(TSB_ForSigSim,[3 1 4 5 6 7 2]);


nTimeSegments=35;
% Time segments
[TSB, Ttimes]=GetTSCoeffsByLinear(nPoints,nTimeSegments);
TSB=permute(TSB,[3 1 4 5 6 7 2]);
TS_ms=Ttimes*nEchoes*EchoSpacing_ms; % times of the time segments


figure;plot(squeeze(TSB));title('Time segments');
%% get signal
ComplexTo3Rows=@(x) [real(x); imag(x); real(x)*0];
Traj3=ComplexTo3Rows(Traj);

SigAllEchoes=bartx('nufft ',Traj3,perm73(IWithSens));

Sig=sum(SigAllEchoes.*TSB_ForSigSim,7); % 1 nPoints 1 nCh
disp('Simulated the signal');
%% get Toeplitz kernel (for each time segment)
% We use [X Y 1 Channels 1 1 TimeSegments LPA]
% LPA is just Linear phase, part of the Toeplitz kernel in BART

Script_NUFFT_ForKernel=[BaseSP 'ForKernel'];
Ops={['nufft 0 1 -1 7 1 0 0 0 ' ToBARTP ]}; % Will save the kernel in ToBARTP
WriteLinopToFile(Script_NUFFT_ForKernel,Ops);

writecfl(SigFN,Sig*0); % Sig is not important for Kernel calculation

delete([lower(ToBARTP) 'LPA*.*']);
delete([lower(ToBARTP) 'PSF*.*']);
clear LPA PSF
for i=1:nTimeSegments
    disp(i);
    CurWeight=TSB(:,:,1,1,1,1,i).^0.5;
    tmp=bartx(['linopScript -d 0 -A ' Script_NUFFT_ForKernel],FillOnesTo16(Sz),SigFN,Traj3,CurWeight);
    PSF(:,:,:,i)=perm74(squeeze(readcfl([lower(ToBARTP) 'PSF'])));
end
LPA=perm74(squeeze(readcfl([lower(ToBARTP) 'LPA'])));

writecfl(LPAFN,perm83(LPA));
writecfl(KernFN,perm83(perm74(real(PSF))));

fgmontagex(log(abs(PSF(:,:,1,:)))) % Makes some sense
%% get A^H (signal). Define the image-to-signal operator using NUFFT
% Define the image-to-signal operator
Ops={   '_Sens fmac 0 0',... % Multiply by sensitivity maps
        '_B0 fmac 1 0',... % Multiply by B0-induced phase evolution, for the time segments
        'nufft 2 -1 -1 7 0 0 0 0',... % Apply NUFFT
        '_TSB fmac 3 64'}; % Sum over the time segments interpolator: 64=2^6 is dim 7
Script_NUFFT_givenB0=[BaseSP 'NUFFT_givenB0'];
WriteLinopToFile(Script_NUFFT_givenB0,Ops);

PhaseDueToB0_TS=perm73(exp(1i*2*pi*perm32(TS_ms).*B0_Hz /1000));

writecfl(SigFN,Sig);
writecfl(B0FN,PhaseDueToB0_TS);
writecfl(TSBFN,TSB);

AHSig=bartx(['linopScript -A -d 0 ' Script_NUFFT_givenB0],FillOnesTo16(Sz),SigFN,SensFN,B0FN,Traj3,TSBFN);
writecfl(AHSigFN,AHSig);

ShowAbsAngle(AHSig) % Still makes sense
%% Now, reconstruct, given B0, using Toeplitz
Ops={   'ident','normal',... % we define the normal operator
        '_Sens fmac 0 0',... % Multiply by sensitivity maps
        '_B0 fmac 1 0',... % Multiply by B0-induced phase evolution, for the time segments
        '_LPA fmac 2 0',... % Linear phase, part of the Toeplitz kernel in BART
        '_FT fft 3',... % Fourier transform over 2D
        'fmac 3 0',... % Apply the Toeplitz kernel
        'a FT','a LPA','a B0','a Sens'};
Script_Toep=[BaseSP 'Toep'];
WriteLinopToFile(Script_Toep,Ops);
ArgsForLinop={SensFN,B0FN,LPAFN,KernFN};

% Rec=bartx(['picsS -w 1 -S -g -m -i 10 -C 3  -d 5  ' Script_Toep],FillOnesTo16(Sz),AHSigFN,ArgsForLinop{:});
Rec=bartx(['picsS -w 1 -S -g -m -R W:3:0:.01 -d 5  ' Script_Toep],FillOnesTo16(Sz),AHSigFN,ArgsForLinop{:});

ShowAbsAngle(cat(3,Rec,I(:,:,1)),1,[0 9]) % Still makes sense
% Doesn't take care of T2* decay!
%% Recon with NUFFT and NUFFT^H can also be done, slowly
% GPU implementation has problem, only CPU works
RecSlow=bartx(['picsS -w 1 -S -m -i 10 -C 3 -d 5 ' Script_NUFFT_givenB0],FillOnesTo16(Sz),SigFN,SensFN,B0FN,Traj3,TSBFN);
ShowAbsAngle(cat(3,RecSlow,I(:,:,1)),1,[0 9]) % Still makes sense
%% That's it
% T=phantom(200);
% TT1=bartx(['linopScript -N -d 0 ' Script_NUFFT_givenB0],FillOnesTo16(Sz),T,SensFN,B0FN,Traj3,TSBFN);
% 
% TT2=bartx(['linopScript -N -d 0 ' Script_Toep],FillOnesTo16(Sz),T,ArgsForLinop{:});
% 
% 
% TWithSens=T.*Sens;
% TWithSensWB0=TWithSens.*PhaseDueToB0_TS;
% clear SigT
% for i=1:nTimeSegments
%     SigT(:,:,:,:,1,1,i)=bartx('nufft ',Traj3,TWithSensWB0(:,:,:,:,1,1,i));
% end
% SigTS=sum(SigT.*TSB,7);
% 
% SigX=SigTS.*conj(TSB);
% % SigX=SigTS;
% clear NSigX
% for i=1:nTimeSegments
%     NSigX(:,:,:,:,1,1,i)=bartx('nufft -a -d 200:200:1 ',Traj3,SigX(:,:,:,:,1,1,i));
% end
% % NSigX=bartx('nufft -a -d 200:200:1 ',ComplexTo3Rows(Traj),SigX);
% NSigXB0=sum(NSigX.*conj(PhaseDueToB0_TS),7);
% AHSigX=sum(NSigXB0.*conj(Sens),4);
% 
% 
% TWithSensWB0WLP=TWithSensWB0.*LPA;
% FTWithSensWB0WLP=fft2(TWithSensWB0WLP);
% K=FTWithSensWB0WLP.*perm74(PSF);
% iK=ifft2(K)*N*N;
% IKL=sum(iK.*conj(LPA),3);
% IKLB=sum(IKL.*conj(PhaseDueToB0_TS),7);
% IKLBS=sum(IKLB.*conj(Sens),4);
% 
% writecfl(SigFN,SigTS);
% Rec=bartx(['picsS -w 1 -S -m -i 10 -C 3 -d 5 ' Script_NUFFT_givenB0],FillOnesTo16(Sz),SigFN,SensFN,B0FN,Traj3,TSBFN);
% 
% 
% Rec=bartx(['picsS -w 1 -S -g -m  -d 5 -i 7 ' Script_Toep],FillOnesTo16(Sz),AHSigX,ArgsForLinop{:});
