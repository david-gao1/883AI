% COMPILE_ALL_ARM64 - Compile all VLFeat MEX files for ARM64
% Run this script in MATLAB to compile all MEX files for ARM64 architecture

fprintf('\n========================================\n');
fprintf('Compiling VLFeat for ARM64 (mexmaca64)\n');
fprintf('========================================\n\n');

% Check architecture
currentArch = mexext;
fprintf('Current MATLAB architecture: %s\n', currentArch);

if ~strcmp(currentArch, 'mexmaca64')
    warning('MATLAB is not running as ARM64. MEX files will be compiled for %s instead.', currentArch);
    targetArch = currentArch;
else
    targetArch = 'mexmaca64';
end

fprintf('Target architecture: %s\n\n', targetArch);

% Get paths
baseDir = fileparts(mfilename('fullpath'));
vlfeatRoot = fullfile(baseDir, 'matchSIFT', 'vlfeat');
toolboxDir = fullfile(vlfeatRoot, 'toolbox');
mexOutDir = fullfile(toolboxDir, 'mex', targetArch);

% Create output directory
if ~exist(mexOutDir, 'dir')
    mkdir(mexOutDir);
    fprintf('Created directory: %s\n', mexOutDir);
end

% Find all C files in toolbox subdirectories
fprintf('\nFinding source files...\n');
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

% Build include flags as cell array (row vector)
includeFlags = {};
for i = 1:length(includeDirs)
    if exist(includeDirs{i}, 'dir')
        includeFlags{1, end+1} = ['-I' includeDirs{i}];
    end
end

% Check for libvl.dylib
libvlPaths = {
    fullfile(vlfeatRoot, 'bin', 'maca64', 'libvl.dylib'),
    fullfile(vlfeatRoot, 'bin', 'maci64', 'libvl.dylib'),
    fullfile(vlfeatRoot, 'bin', 'maci', 'libvl.dylib')
};

libvlPath = '';
for i = 1:length(libvlPaths)
    if exist(libvlPaths{i}, 'file')
        libvlPath = libvlPaths{i};
        break;
    end
end

if ~isempty(libvlPath)
    fprintf('Found libvl.dylib at: %s\n', libvlPath);
    % Copy to MEX directory
    copyfile(libvlPath, fullfile(mexOutDir, 'libvl.dylib'));
    fprintf('Copied libvl.dylib to MEX directory\n');
    libDir = fileparts(libvlPath);
else
    warning('libvl.dylib not found. MEX files may fail to link.');
    libDir = '';
end

fprintf('\n========================================\n');
fprintf('Compiling MEX files...\n');
fprintf('========================================\n\n');

successCount = 0;
failCount = 0;
failedFiles = {};

for i = 1:length(srcFiles)
    srcFile = srcFiles{i};
    [~, name, ~] = fileparts(srcFile);
    mexFile = fullfile(mexOutDir, [name '.' targetArch]);
    
    fprintf('[%d/%d] %s\n', i, length(srcFiles), name);
    
    try
        % Build MEX command arguments one by one
        % Start with basic flags
        mexArgs = {['-' targetArch], '-largeArrayDims', '-O'};
        
        % Add include directories
        for k = 1:length(includeFlags)
            mexArgs{end+1} = includeFlags{k};
        end
        
        % Add source file and output directory
        mexArgs{end+1} = srcFile;
        mexArgs{end+1} = '-outdir';
        mexArgs{end+1} = mexOutDir;
        
        % Add library if available
        if ~isempty(libDir)
            mexArgs{end+1} = ['-L' libDir];
            mexArgs{end+1} = '-lvl';
        end
        
        % Compile - pass all arguments
        mex(mexArgs{:});
        
        % Verify
        if exist(mexFile, 'file')
            fprintf('  ✓ Success\n');
            successCount = successCount + 1;
        else
            fprintf('  ✗ Warning: File not created\n');
            failCount = failCount + 1;
            failedFiles{end+1} = name;
        end
    catch ME
        fprintf('  ✗ Error: %s\n', ME.message);
        failCount = failCount + 1;
        failedFiles{end+1} = name;
    end
end

fprintf('\n========================================\n');
fprintf('Compilation Summary\n');
fprintf('========================================\n');
fprintf('Success: %d\n', successCount);
fprintf('Failed:  %d\n', failCount);

if failCount > 0
    fprintf('\nFailed files:\n');
    for i = 1:length(failedFiles)
        fprintf('  - %s\n', failedFiles{i});
    end
end

fprintf('\n========================================\n');

if successCount > 0
    fprintf('✓ Compilation completed!\n');
    fprintf('MEX files are in: %s\n', mexOutDir);
    fprintf('\nYou can now run SFMedu2.m\n');
else
    fprintf('✗ Compilation failed!\n');
    fprintf('Please check errors above.\n');
end

fprintf('========================================\n\n');

