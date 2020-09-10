function channelsensitivityOdd=RunESPIRiTForSensMapsMultiMap(Data,rVal,Sz,k)
FF=squeeze(bart('fft 6',permute(Data,[4 1 2 3])));
if(rVal==0)
    RStr=' -a ';
else
    RStr=[' -r ' num2str(rVal) ' '];
end
if(nargin<4)
    KStr='';
else
    KStr=[' -k ' num2str(k) ' '];
end

if(nargin>2)
    if(Sz(1)<size(FF,1))
        FF=crop(FF,Sz(1),size(FF,2),size(FF,3),size(FF,4));
    end
    if(Sz(2)<size(FF,2))
        FF=crop(FF,size(FF,1),Sz(2),size(FF,3),size(FF,4));
    end
    if(Sz(1)>size(FF,1))
        FF=padBoth(FF,(Sz(1)-size(FF,1))/2,1);
    end
    if(Sz(2)>size(FF,2))
        FF=padBoth(FF,(Sz(2)-size(FF,2))/2,2);
    end
    FFs(1,:,:,:)=FF;
    
else
    FFs(1,:,:,:)=FF;
end
calib = bart(['ecalib ' RStr KStr], FFs);
channelsensitivityOdd=squeeze(calib);
% channelsensitivityOdd(:,:,:,1) = permute(bart('slice 4 0', calib),[2 3 4 1]);
% ShowAbsAngle(channelsensitivityOdd(:,:,:,1));
