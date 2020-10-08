function [varargout] = bart(cmd, varargin)
% BART	Call BART command from Matlab.
%   [A, B] = bart('command', X, Y) call command with inputs X Y and outputs A B
%
% 2014-2015 Martin Uecker <uecker@med.uni-goettingen.de>

% setenv('TOOLBOX_PATH','/autofs/cluster/kawin/sid/bart')
% setenv('SHELL','/bin/bash')
% setenv('MATLAB_SHELL','/bin/bash')
% system('/bin/bash /usr/pubsw/packages/mkl/2019/bin/compilervars.sh intel64')

	if nargin==0
		disp('Usage: bart <command> <arguments...>');
		return
    end
    if cmd==0
		disp('Usage: bart <command> <arguments...>');
		return
    end
    
	bart_path = getenv('TOOLBOX_PATH');

	if isempty(bart_path)
		if exist('/usr/local/bin/bart', 'file')
			bart_path = '/usr/local/bin';
		elseif exist('/usr/bin/bart', 'file')
			bart_path = '/usr/bin';
		else
			error('Environment variable TOOLBOX_PATH is not set.');
		end
	end

	% clear the LD_LIBRARY_PATH environment variable (to work around
	% a bug in Matlab).

	if ismac==1
		setenv('DYLD_LIBRARY_PATH', '');
	else
% 		setenv('LD_LIBRARY_PATH', '');
        setenv('LD_LIBRARY_PATH','/usr/pubsw/packages/mkl/2019/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64_lin:/autofs/cluster/pubsw/arch/CentOS7-x86_64/packages/mkl/2019//compilers_and_libraries_2019.0.117/linux/mpi/intel64/lib/release:/autofs/cluster/pubsw/arch/CentOS7-x86_64/packages/mkl/2019//compilers_and_libraries_2019.0.117/linux/mpi/intel64/lib:/usr/pubsw/packages/mkl/2019/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64_lin:/usr/pubsw/packages/mkl/2019/compilers_and_libraries_2019.0.117/linux/mkl/lib/intel64_lin:/autofs/cluster/pubsw/arch/CentOS7-x86_64/packages/mkl/2019/compilers_and_libraries_2019.0.117/linux/tbb/lib/intel64/gcc4.7:/autofs/cluster/pubsw/arch/CentOS7-x86_64/packages/mkl/2019/compilers_and_libraries_2019.0.117/linux/tbb/lib/intel64/gcc4.7');
    end

    TempPath=getenv('TEMP_PATH');
    if(isempty(TempPath))
        name = tempname;
    else
%         name = tempname('/local_mount/space/yi/1/users/Gilad/tmp/');
        name = tempname(TempPath);
    end
    
	in = cell(1, nargin - 1);

    CharB=false(1,nargin - 1);
	for i=1:(nargin - 1)
        if(ischar(varargin{i}))
            CharB(i)=true;
            in{i}=varargin{i};
        else
            in{i} = strcat(name, 'in', num2str(i));
            writecfl(in{i}, varargin{i});
        end
	end

	in_str = sprintf(' %s', in{:});

	out = cell(1, nargout);

	for i=1:nargout,
		out{i} = strcat(name, 'out', num2str(i));
	end

	out_str = sprintf(' %s', out{:});

	if ispc
		% For cygwin use bash and modify paths
		ERR = system(['C:\cygwin64\bin\bash.exe --login -c ', ...
			strrep(bart_path, filesep, '/'), ...
	                '"', '/bart ', strrep(cmd, filesep, '/'), ' ', ...
			strrep(in_str, filesep, '/'), ...
                	' ', strrep(out_str, filesep, '/'), '"']);
    else
%         if(strcmp(strtok(cmd),'linopScript') || strcmp(strtok(cmd),'picsS'))
%         if(strcmp(strtok(cmd),'picsS'))
%             disp([bart_path, '/bart ', cmd, ' ', out_str, ' ', in_str]);
%             ERR = system([bart_path, '/bart ', cmd, ' ', out_str, ' ', in_str]);
%         else
            ERR = system([bart_path, '/bart ', cmd, ' ', in_str, ' ', out_str]);
%         end
	end

	for i=1:(nargin - 1)
        if(~CharB(i))
            if (exist(strcat(in{i}, '.cfl'),'file'))
                delete(strcat(in{i}, '.cfl'));
            end

            if (exist(strcat(in{i}, '.hdr'),'file'))
                delete(strcat(in{i}, '.hdr'));
            end
        end
	end

	for i=1:nargout,
		if ERR==0
			varargout{i} = readcfl(out{i});
		end
		if (exist(strcat(out{i}, '.cfl'),'file'))
			delete(strcat(out{i}, '.cfl'));
		end
		if (exist(strcat(out{i}, '.hdr'),'file'))
			delete(strcat(out{i}, '.hdr'));
		end
	end

	if ERR~=0
		error('command exited with an error');
	end
end
