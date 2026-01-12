% COMPILE_ARM64 - Compile VLFeat MEX files for ARM64 (Apple Silicon)
% This script compiles all VLFeat MEX files for ARM64 architecture

fprintf('========================================\n');
fprintf('Compiling VLFeat for ARM64 (mexmaca64)\n');
fprintf('========================================\n\n');

% Get current directory and VLFeat path
currentDir = pwd;
vlfeatRoot = fullfile(currentDir, 'matchSIFT', 'vlfeat');
toolboxDir = fullfile(vlfeatRoot, 'toolbox');
mexDir = fullfile(toolboxDir, 'mex', 'mexmaca64');

% Create output directory
if ~exist(mexDir, 'dir')
    mkdir(mexDir);
    fprintf('Created directory: %s\n', mexDir);
end

% Check MATLAB architecture
fprintf('Current MATLAB architecture: %s\n', mexext);
fprintf('Target architecture: mexmaca64\n\n');

if ~strcmp(mexext, 'mexmaca64')
    warning('MATLAB is not running as ARM64. MEX files will be compiled for current architecture.');
end

% Find all C source files in toolbox subdirectories
fprintf('Finding source files...\n');
subdirs = {'aib', 'geometry', 'imop', 'kmeans', 'misc', 'mser', 'plotop', ...
           'quickshift', 'sift', 'slic'};

allCFiles = {};
for i = 1:length(subdirs)
    subdir = fullfile(toolboxDir, subdirs{i});
    if exist(subdir, 'dir')
        files = dir(fullfile(subdir, '*.c'));
        for j = 1:length(files)
            allCFiles{end+1} = fullfile(subdir, files(j).name);
        end
    end
end

fprintf('Found %d C source files\n\n', length(allCFiles));

% Compile options
includeDirs = {
    vlfeatRoot,
    toolboxDir,
    fullfile(vlfeatRoot, 'vl')
};

% Build include flags
includeFlags = {};
for i = 1:length(includeDirs)
    includeFlags{end+1} = ['-I' includeDirs{i}];
end

% Check if libvl.dylib exists (needed for linking)
libvlPath = fullfile(vlfeatRoot, 'bin', 'maci64', 'libvl.dylib');
if ~exist(libvlPath, 'file')
    % Try alternative location
    libvlPath = fullfile(vlfeatRoot, 'bin', 'maci', 'libvl.dylib');
    if ~exist(libvlPath, 'file')
        warning('libvl.dylib not found. MEX files may fail to link.');
        libvlPath = '';
    end
end

if ~isempty(libvlPath)
    fprintf('Found libvl.dylib at: %s\n', libvlPath);
    % Copy to MEX directory
    copyfile(libvlPath, fullfile(mexDir, 'libvl.dylib'));
    fprintf('Copied libvl.dylib to MEX directory\n\n');
end

% Compile each MEX file
fprintf('Compiling MEX files...\n');
fprintf('========================================\n');

successCount = 0;
failCount = 0;

for i = 1:length(allCFiles)
    srcFile = allCFiles{i};
    [~, name, ~] = fileparts(srcFile);
    mexFile = fullfile(mexDir, [name '.mexmaca64']);
    
    fprintf('[%d/%d] Compiling %s...\n', i, length(allCFiles), name);
    
    try
        % Build MEX command
        mexCmd = {
            '-mexmaca64',           % Target architecture
            '-largeArrayDims',      % Use 64-bit array indexing
            '-O',                   % Optimize
            includeFlags{:},        % Include directories
            srcFile,                % Source file
            '-outdir', mexDir       % Output directory
        };
        
        % Add library path if available
        if ~isempty(libvlPath)
            libDir = fileparts(libvlPath);
            mexCmd{end+1} = ['-L' libDir];
            mexCmd{end+1} = '-lvl';
        end
        
        % Compile
        mex(mexCmd{:});
        
        if exist(mexFile, 'file')
            fprintf('  ✓ Success: %s\n', mexFile);
            successCount = successCount + 1;
        else
            fprintf('  ✗ Warning: MEX file not created\n');
            failCount = failCount + 1;
        end
    catch ME
        fprintf('  ✗ Error: %s\n', ME.message);
        failCount = failCount + 1;
    end
    fprintf('\n');
end

fprintf('========================================\n');
fprintf('Compilation Summary:\n');
fprintf('  Success: %d\n', successCount);
fprintf('  Failed:  %d\n', failCount);
fprintf('========================================\n');

if successCount > 0
    fprintf('\n✓ Compilation completed! MEX files are in:\n');
    fprintf('  %s\n', mexDir);
    fprintf('\nYou can now run SFMedu2.m\n');
else
    fprintf('\n✗ Compilation failed. Please check errors above.\n');
end

