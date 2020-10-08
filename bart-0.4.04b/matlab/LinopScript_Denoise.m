addpath('/autofs/space/daisy_002/users/Gilad/bart-0.4.04b/matlab/');

setenv('TOOLBOX_PATH','/autofs/space/daisy_002/users/Gilad/bart-0.4.04b/');

BaseSP='/autofs/space/daisy_002/users/Gilad';

setenv('TEMP_PATH','/dev/shm/');

ToBARTP=['/dev/shm/LS_' num2str(floor(rand*1e10)) '_'];
%% Operation: identity (do nothing)
Ops={'ident'};
ScriptFN=[BaseSP 'ident.txt'];
WriteLinopToFile(ScriptFN,Ops);
%%
I=phantom(200);

Sz16=FillOnesTo16(size(I));
% Apply the operator
I_Copy=bart(['linopScript ' ScriptFN],Sz16,I);

% They are the same
mean(abs(I-I_Copy),'all')
%%
I_Noised=I+randn(size(I))*0.03;

% Clean the image by ADMM + TV regularization
I_Cleaned=bartx(['picsS -S -R T:3:0:.01 ' ScriptFN],Sz16,I_Noised);
%% Look at the results
fgmontagex(cat(3,I,I_Noised,I_Cleaned),'Size',[1 3])
% Nice