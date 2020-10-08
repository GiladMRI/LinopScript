function Out=FillOnesTo16(In)
Out=[Col(In); ones(16-numel(In),1)];