function path = vl_setup(varargin)
% VL_SETUP Add VLFeat Toolbox to the path
%   PATH = VL_SETUP() adds the VLFeat Toolbox to MATLAB path and
%   returns the path PATH to the VLFeat package.
%
%   VL_SETUP('NOPREFIX') adds aliases to each function that do not
%   contain the VL_ prefix. For example, with this option it is
%   possible to use SIFT() instead of VL_SIFT().
%
%   VL_SETUP('TEST') or VL_SETUP('XTEST') adds VLFeat unit test
%   function suite. See also VL_TEST().
%
%   VL_SETUP('QUIET') does not print the greeting message.
%
%   See also: VL_ROOT(), VL_HELP().

% Authors: Andrea Vedaldi and Brian Fulkerson

% Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

noprefix = false ;
quiet = true ;
xtest = false ;
demo = false ;

for ai=1:length(varargin)
  opt = varargin{ai} ;
  switch lower(opt)
    case {'noprefix', 'usingvl'}
      noprefix = true ;
    case {'test', 'xtest'}
      xtest = true ;
    case {'demo'}
      demo = true ;
    case {'quiet'}
      quiet = true ;
    case {'verbose'}
      quiet = false ;
    otherwise
      error('Unknown option ''%s''.', opt) ;
  end
end

if exist('octave_config_info')
  bindir = 'octave' ;
else
  bindir = mexext ;
  if strcmp(bindir, 'dll'), bindir = 'mexw32' ; end
end

% Handle architecture compatibility: if requested MEX directory doesn't exist,
% try to fall back to compatible alternatives
[a,b,c] = fileparts(mfilename('fullpath')) ;
[a,b,c] = fileparts(a) ;
root = a ;
mexdir = fullfile(root,'toolbox','mex',bindir) ;

% Check if MEX directory exists and has MEX files
% If directory exists but has no MEX files, try fallback
mexDirValid = false;
if exist(mexdir, 'dir')
  % Check if directory has any MEX files
  mexFiles = dir(fullfile(mexdir, ['*.' bindir]));
  if ~isempty(mexFiles)
    mexDirValid = true;
  end
end

% If the requested MEX directory doesn't exist or has no MEX files, try fallback options
if ~mexDirValid
  % Try fallback architectures (in order of preference)
  fallbackDirs = {};
  if strcmp(bindir, 'mexmaca64')
    % ARM64 Mac: try x86_64 (Rosetta 2 compatible)
    fallbackDirs = {'mexmaci64', 'mexmaci'};
  elseif strcmp(bindir, 'mexmaci64')
    % x86_64 Mac: try 32-bit as fallback
    fallbackDirs = {'mexmaci'};
  end
  
  % Try fallback directories
  foundFallback = false;
  for i = 1:length(fallbackDirs)
    testDir = fullfile(root,'toolbox','mex',fallbackDirs{i});
    if exist(testDir, 'dir')
      % Check if fallback directory has MEX files
      fallbackMexFiles = dir(fullfile(testDir, ['*.' fallbackDirs{i}]));
      if ~isempty(fallbackMexFiles)
        bindir = fallbackDirs{i};
        mexdir = testDir;
        foundFallback = true;
        if ~quiet
          fprintf('Note: Using %s MEX files instead of %s.\n', bindir, mexext);
        end
        break;
      end
    end
  end
  
  % If no fallback found, this will be caught later when adding path
end

% Build the full path to mex directory
bindir = fullfile('mex',bindir) ;

% Do not use vl_root() to avoid conflicts with other VLFeat
% installations.
% (root variable already set above for architecture check)

% IMPORTANT: Add MEX directory FIRST to ensure MEX files are found before .m files
% root is already absolute path from mfilename('fullpath')
mexFullPath = fullfile(root,'toolbox',bindir);
if ~exist(mexFullPath, 'dir')
    error('VLFeat MEX directory not found: %s', mexFullPath);
end

% Add path and clear cache
addpath(mexFullPath, '-begin') ;

% Clear function cache to ensure MEX files are loaded
clear('vl_sift', 'vl_version');

% Force MATLAB to rehash the path
rehash path;

addpath(fullfile(root,'toolbox'             )) ;
addpath(fullfile(root,'toolbox','aib'       )) ;
addpath(fullfile(root,'toolbox','geometry'  )) ;
addpath(fullfile(root,'toolbox','imop'      )) ;
addpath(fullfile(root,'toolbox','kmeans'    )) ;
addpath(fullfile(root,'toolbox','misc'      )) ;
addpath(fullfile(root,'toolbox','mser'      )) ;
addpath(fullfile(root,'toolbox','plotop'    )) ;
addpath(fullfile(root,'toolbox','quickshift')) ;
addpath(fullfile(root,'toolbox','sift'      )) ;
addpath(fullfile(root,'toolbox','special'   )) ;
addpath(fullfile(root,'toolbox','slic'      )) ;

if noprefix
  addpath(fullfile(root,'toolbox','noprefix')) ;
end

if xtest
  addpath(fullfile(root,'toolbox','xtest')) ;
end

if demo
  addpath(fullfile(root,'toolbox','demo')) ;
end

if ~quiet
  if exist('vl_version') == 3
    fprintf('VLFeat %s ready.\n', vl_version) ;
  else
    warning('VLFeat does not seem to be installed correctly. Make sure that the MEX files are compiled.') ;
  end
end

% Verify that MEX files are accessible (always check, even if quiet)
% Force MATLAB to refresh its function cache
rehash toolbox;
pause(0.2);  % Give MATLAB more time to process

if exist('vl_sift', 'file') ~= 3
    warning('vl_sift MEX file not found. MEX path: %s. Current mexext: %s', mexFullPath, mexext);
    
    % Check if MEX file actually exists
    mexFile = fullfile(mexFullPath, 'vl_sift.mexmaci64');
    if exist(mexFile, 'file')
        fprintf('MEX file exists: %s\n', mexFile);
        fprintf('File info: ');
        dir(mexFile);
        
        % Try direct load
        try
            % Attempt to load the MEX file directly
            [pathstr, name, ext] = fileparts(mexFile);
            oldDir = cd(pathstr);
            try
                eval(sprintf('clear %s;', name));
                eval(sprintf('which %s', name));
            catch
            end
            cd(oldDir);
        catch ME2
            fprintf('Direct load failed: %s\n', ME2.message);
        end
    else
        fprintf('MEX file not found at: %s\n', mexFile);
        % List all files in directory
        allFiles = dir(mexFullPath);
        fprintf('Files in MEX directory:\n');
        for i = 1:min(10, length(allFiles))
            fprintf('  %s\n', allFiles(i).name);
        end
    end
    
    % Architecture mismatch warning
    if strcmp(mexext, 'mexmaca64')
        warning(['MATLAB is running as ARM64 (mexmaca64) but MEX files are x86_64 (mexmaci64). ' ...
            'You may need to run MATLAB under Rosetta 2, or recompile MEX files for ARM64.']);
    end
end

if nargout == 0
  clear path ;
end
