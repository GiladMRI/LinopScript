addpath('/autofs/space/daisy_002/users/Gilad/bart-0.4.04b/matlab/');

setenv('TOOLBOX_PATH','/autofs/space/daisy_002/users/Gilad/bart-0.4.04b/');

BaseSP='/autofs/space/daisy_002/users/Gilad';

setenv('TEMP_PATH','/dev/shm/');

ToBARTP=['/dev/shm/LS_' num2str(floor(rand*1e10)) '_'];

SampMaskFN=[ToBARTP 'Msk'];
SensFN=[ToBARTP 'Sens'];
CompsFN=[ToBARTP 'Comps'];
B0FN=[ToBARTP 'B0'];

AHSigFN=[ToBARTP 'AHSig'];
%% load senstivity maps and quantitative maps (PD phase, angle, B0, R2, and R2*)
load('ExampleSensAnsMaps.mat');
%% Simulate data
EchoSpacing_ms=1;

nEchoes=40;
TE0_ms=8;
TEs_ms=TE0_ms+(0:nEchoes-1)*EchoSpacing_ms;

SliceIdx=10;

PD=-MapsGESES(:,:,1,SliceIdx).*exp(1i*MapsGESES(:,:,2,SliceIdx));
PhaseDueToB0=exp(1i*2*pi*perm32(TEs_ms).*MapsGESES(:,:,3,SliceIdx)/1000);
DecayDueToT2star=exp(-perm32(TEs_ms).*MapsGESES(:,:,5,SliceIdx));
I=PD.*PhaseDueToB0.*DecayDueToT2star;

IWithSens=I.*SensPCS(:,:,SliceIdx,:);
nPE=size(I,1);
nRO=size(I,2);
% IWithSens [PE, RO, Echoes, Channels]

%% sampling mask
SampMask=zeros([nPE 1 nEchoes]);
nShots=9;
RSeg=floor(nPE/nShots);
R=3;
for i=1:nEchoes
    SampMask( mod(i*R+ RSeg*(0:nShots-1) ,nPE)+1  ,1,i)=1;
end
% fgmontagex(squeeze(SampMask))
kData=fft2cg(IWithSens).*SampMask;
%% Subspace components
I2d=reshape(abs(I),[],nEchoes);
[~,~,AllComps]=svd(I2d,'econ');

nCompsToUse=3;
Comps=AllComps(:,1:nCompsToUse);
% figure;plot(Comps)
clear I2d

CompsP=repmat(permute(Comps,[3 4 1 5 6 2]),[nPE nRO]);
% CompsP: [PE RO Echoes 1 Comps]
% repmat for better GPU performance in BART (fixed?)
%% Define the operator by linopScript
% EPTI using subspace given B0
% From image space: Open components. Apply sensitivity maps. Apply B0 effect. Do FT on PE. Apply sampling mask
% Open components: Multiply and sum over the components dimension, e.g. Dim 6, BART flag 32
Ops={'_Comps fmac 0 32','_Sens fmac 1 0','_B0 fmac 2 0','_FT fftc 1','_Samp fmac 3 0'};
ScriptFN=[BaseSP 'EPTI.txt'];
WriteLinopToFile(ScriptFN,Ops);
ArgsForLinop={CompsFN,SensFN,B0FN,SampMaskFN};

writecfl(SensFN,SensPCS(:,:,SliceIdx,:));
writecfl(CompsFN,CompsP);
writecfl(B0FN,PhaseDueToB0);
writecfl(SampMaskFN,repmat(SampMask,[1 nRO 1 1]));

% no need to do FTs on the fully sampled RO
kData_IRO=ifft1cg(kData,2);

Sz16=FillOnesTo16([nPE nRO 1 1 1 nCompsToUse]);
%%
Rec=bartx(['picsS -g -w 1 -S -m  -R L:3:3:0.01 -i 100 -d 5 ' ScriptFN],Sz16,kData_IRO,ArgsForLinop{:});
RecEchoes=sum(Rec.*CompsP.*PhaseDueToB0,6);

ShowAbsAngle(flipud(RecEchoes(:,:,3:7:end))) % Nice
%% Separate AHsig calculation and the optimization: Helps mainly for non cartesian where
% AHsig calculation is heavy
% get the A^H sig
AHsig=bartx(['linopScript -A -d 5 ' ScriptFN],Sz16,kData_IRO,ArgsForLinop{:});
writecfl(AHSigFN,AHsig);
%% The normal script: can be customized when subspace spatiotemporal trick is used
% (not in EPTI due to B0 contribution)
Ops={'ident','normal','_Comps fmac 0 32','_Sens fmac 1 0','_B0 fmac 2 0','_FT fftc 1','_Samp fmac 3 0',...
    'a FT','a B0','a Sens','a Comps'}; % a means the adjoint of the corresponding operation

ScriptFN_Normal=[BaseSP 'EPTIN.txt'];
WriteLinopToFile(ScriptFN_Normal,Ops);

RecN=bartx(['picsS -g -w 1 -S -m  -R L:3:3:0.01 -i 100 -d 5 ' ScriptFN_Normal],Sz16,AHSigFN,ArgsForLinop{:});
RecEchoesN=sum(RecN.*CompsP.*PhaseDueToB0,6);
%%
ShowAbsAngle(flipud(RecEchoesN(:,:,3:7:end))) % Same
%% Now, do the calculation per 2 channels, to save memory