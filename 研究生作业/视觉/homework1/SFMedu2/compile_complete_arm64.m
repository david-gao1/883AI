% COMPILE_COMPLETE_ARM64 - Complete compilation of VLFeat for ARM64
% This script compiles both the library and MEX files for ARM64

fprintf('\n========================================\n');
fprintf('Complete VLFeat ARM64 Compilation\n');
fprintf('========================================\n\n');

% Check architecture
currentArch = mexext;
fprintf('Current MATLAB architecture: %s\n', currentArch);

if ~strcmp(currentArch, 'mexmaca64')
    warning('MATLAB is not running as ARM64. Results may not work correctly.');
end

% Get paths
baseDir = fileparts(mfilename('fullpath'));
vlfeatRoot = fullfile(baseDir, 'matchSIFT', 'vlfeat');
cd(vlfeatRoot);

fprintf('VLFeat root: %s\n\n', vlfeatRoot);

% Step 1: Compile libvl.dylib using Makefile
fprintf('========================================\n');
fprintf('Step 1: Compiling libvl.dylib\n');
fprintf('========================================\n\n');

% Use system command to run make
% DISABLE_SSE2=yes is automatically set in Makefile for maca64
makeCmd = sprintf('make ARCH=maca64 clean 2>&1');
fprintf('Running: make ARCH=maca64 clean\n');
[status, output] = system(makeCmd);
if status ~= 0
    fprintf('Clean output: %s\n', output);
end

makeCmd = sprintf('make ARCH=maca64 DISABLE_SSE2=yes 2>&1');
fprintf('Running: make ARCH=maca64 DISABLE_SSE2=yes\n');
fprintf('This may take a few minutes...\n\n');
[status, output] = system(makeCmd);

if status ~= 0
    fprintf('Make output:\n%s\n', output);
    error('Failed to compile libvl.dylib. Check errors above.');
end

fprintf('✓ libvl.dylib compiled successfully\n\n');

% Step 2: Verify libvl.dylib exists
libvlPath = fullfile(vlfeatRoot, 'bin', 'maca64', 'libvl.dylib');
if ~exist(libvlPath, 'file')
    error('libvl.dylib not found at: %s', libvlPath);
end

% Step 3: Compile MEX files
fprintf('========================================\n');
fprintf('Step 2: Compiling MEX files\n');
fprintf('========================================\n\n');

toolboxDir = fullfile(vlfeatRoot, 'toolbox');
mexOutDir = fullfile(toolboxDir, 'mex', 'mexmaca64');

if ~exist(mexOutDir, 'dir')
    mkdir(mexOutDir);
end

% Copy libvl.dylib to MEX directory
copyfile(libvlPath, fullfile(mexOutDir, 'libvl.dylib'));
fprintf('Copied libvl.dylib to MEX directory\n\n');

% Find all C files
subdirs = {'aib', 'geometry', 'imop', 'kmeans', 'misc', 'mser', 'plotop', ...
           'quickshift', 'sift', 'slic'};

srcFiles = {};
for i = 1:length(subdirs)
    subdir = fullfile(toolboxDir, subdirs{i});
    if exist(subdir, 'dir')
        files = dir(fullfile(subdir, '*.c'));
        for j = 1:length(files)
            srcFiles{end+1} = fullfile(subdir, files(j).name);
        end
    end
end

fprintf('Found %d C source files\n\n', length(srcFiles));

% Include directories
includeDirs = {
    vlfeatRoot,
    toolboxDir,
    fullfile(vlfeatRoot, 'vl')
};

% Build include flags
includeFlags = {};
for i = 1:length(includeDirs)
    if exist(includeDirs{i}, 'dir')
        includeFlags{1, end+1} = ['-I' includeDirs{i}];
    end
end

libDir = fileparts(libvlPath);

% Compile each MEX file
successCount = 0;
failCount = 0;
failedFiles = {};

for i = 1:length(srcFiles)
    srcFile = srcFiles{i};
    [~, name, ~] = fileparts(srcFile);
    mexFile = fullfile(mexOutDir, [name '.mexmaca64']);
    
    fprintf('[%d/%d] %s', i, length(srcFiles), name);
    
    try
        % Build MEX command
        % Note: Don't specify -mexmaca64, MATLAB automatically uses current architecture
        mexArgs = {'-largeArrayDims', '-O'};
        
        % Add include directories
        for k = 1:length(includeFlags)
            mexArgs{end+1} = includeFlags{k};
        end
        
        % Add source file and output directory
        mexArgs{end+1} = srcFile;
        mexArgs{end+1} = '-outdir';
        mexArgs{end+1} = mexOutDir;
        
        % Add library
        mexArgs{end+1} = ['-L' libDir];
        mexArgs{end+1} = '-lvl';
        
        % Compile
        mex(mexArgs{:});
        
        % Verify
        if exist(mexFile, 'file')
            fprintf(' ✓\n');
            successCount = successCount + 1;
        else
            fprintf(' ✗ (file not created)\n');
            failCount = failCount + 1;
            failedFiles{end+1} = name;
        end
    catch ME
        fprintf(' ✗ (%s)\n', ME.message);
        failCount = failCount + 1;
        failedFiles{end+1} = name;
    end
end

fprintf('\n========================================\n');
fprintf('Compilation Summary\n');
fprintf('========================================\n');
fprintf('Success: %d\n', successCount);
fprintf('Failed:  %d\n', failCount);

if failCount > 0 && failCount < length(srcFiles)
    fprintf('\nSome files failed, but compilation may still be usable.\n');
    fprintf('Failed files:\n');
    for i = 1:min(10, length(failedFiles))
        fprintf('  - %s\n', failedFiles{i});
    end
    if length(failedFiles) > 10
        fprintf('  ... and %d more\n', length(failedFiles) - 10);
    end
end

fprintf('\n========================================\n');

if successCount > 0
    fprintf('✓ Compilation completed!\n');
    fprintf('MEX files are in: %s\n', mexOutDir);
    fprintf('\nYou can now run:\n');
    fprintf('  vl_setup(''quiet'')\n');
    fprintf('  SFMedu2\n');
else
    fprintf('✗ Compilation failed!\n');
    fprintf('Please check errors above.\n');
end

fprintf('========================================\n\n');

