% COMPILE_MEX_ONLY_ARM64 - Compile only MEX files for ARM64
% This script compiles MEX files directly in MATLAB without needing the library

fprintf('\n========================================\n');
fprintf('Compiling VLFeat MEX files for ARM64\n');
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
fprintf('Finding source files...\n');
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

% Try to find libvl.dylib (use x86_64 version if ARM64 not available)
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

% Add VL_DISABLE_SSE2 define for ARM64
if strcmp(targetArch, 'mexmaca64')
    includeFlags{1, end+1} = '-DVL_DISABLE_SSE2';
    fprintf('Added -DVL_DISABLE_SSE2 for ARM64\n');
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
    
    fprintf('[%d/%d] %s', i, length(srcFiles), name);
    
    try
        % Build MEX command arguments
        % Note: MATLAB's mex command doesn't use -mexmaca64 format
        % Instead, we specify the output file extension directly
        mexArgs = {'-largeArrayDims', '-O'};
        
        % Add include directories
        for k = 1:length(includeFlags)
            mexArgs{end+1} = includeFlags{k};
        end
        
        % Add source file and output directory with extension
        mexArgs{end+1} = srcFile;
        mexArgs{end+1} = '-outdir';
        mexArgs{end+1} = mexOutDir;
        
        % Add library if available
        if ~isempty(libDir)
            mexArgs{end+1} = ['-L' libDir];
            mexArgs{end+1} = '-lvl';
        end
        
        % Compile - MATLAB will automatically use the correct architecture
        % based on the current MATLAB installation (no need to specify -mexmaca64)
        mex(mexArgs{:});
        
        % Check what file was actually created (MATLAB auto-detects architecture)
        % Look for any .mex* file with this name
        possibleExtensions = {'mexmaca64', 'mexmaci64', 'mexmaci', 'mexa64', 'mexglx'};
        foundFile = false;
        actualExt = '';
        createdFile = '';
        
        for extIdx = 1:length(possibleExtensions)
            testFile = fullfile(mexOutDir, [name '.' possibleExtensions{extIdx}]);
            if exist(testFile, 'file')
                actualExt = possibleExtensions{extIdx};
                createdFile = testFile;
                foundFile = true;
                break;
            end
        end
        
        % If file was created with different extension, rename it to target
        if foundFile && ~strcmp(actualExt, targetArch)
            targetFile = fullfile(mexOutDir, [name '.' targetArch]);
            if ~exist(targetFile, 'file')
                movefile(createdFile, targetFile);
                createdFile = targetFile;
            end
        end
        
        % Verify
        if foundFile
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
    if length(failedFiles) <= 10
        fprintf('Failed files:\n');
        for i = 1:length(failedFiles)
            fprintf('  - %s\n', failedFiles{i});
        end
    else
        fprintf('Failed files (first 10):\n');
        for i = 1:10
            fprintf('  - %s\n', failedFiles{i});
        end
        fprintf('  ... and %d more\n', length(failedFiles) - 10);
    end
end

fprintf('\n========================================\n');

if successCount > 0
    fprintf('✓ Compilation completed!\n');
    fprintf('MEX files are in: %s\n', mexOutDir);
    
    % Test if vl_sift can be loaded
    fprintf('\nTesting vl_sift...\n');
    clear('vl_sift');
    rehash path;
    pause(0.2);
    
    if exist('vl_sift', 'file') == 3
        fprintf('✓ vl_sift MEX file loaded successfully!\n');
        fprintf('\nYou can now run:\n');
        fprintf('  vl_setup(''quiet'')\n');
        fprintf('  SFMedu2\n');
    else
        fprintf('⚠ vl_sift not found. You may need to run vl_setup again.\n');
    end
else
    fprintf('✗ Compilation failed!\n');
    fprintf('Please check errors above.\n');
end

fprintf('========================================\n\n');

