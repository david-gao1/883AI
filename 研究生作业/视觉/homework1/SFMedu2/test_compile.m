% TEST_COMPILE - Test compiling a single MEX file to debug issues

fprintf('Testing MEX compilation...\n\n');

% Get paths
baseDir = fileparts(mfilename('fullpath'));
vlfeatRoot = fullfile(baseDir, 'matchSIFT', 'vlfeat');
toolboxDir = fullfile(vlfeatRoot, 'toolbox');
mexOutDir = fullfile(toolboxDir, 'mex', 'mexmaca64');

% Test with vl_sift
srcFile = fullfile(toolboxDir, 'sift', 'vl_sift.c');

fprintf('Source file: %s\n', srcFile);
fprintf('Output directory: %s\n', mexOutDir);
fprintf('Architecture: mexmaca64\n\n');

% Include directories
includeDirs = {
    vlfeatRoot,
    toolboxDir,
    fullfile(vlfeatRoot, 'vl')
};

% Build include flags one by one
fprintf('Include directories:\n');
for i = 1:length(includeDirs)
    fprintf('  -I%s\n', includeDirs{i});
end

% Library path
libvlPath = fullfile(vlfeatRoot, 'bin', 'maca64', 'libvl.dylib');
if ~exist(libvlPath, 'file')
    libvlPath = fullfile(vlfeatRoot, 'bin', 'maci64', 'libvl.dylib');
end

if exist(libvlPath, 'file')
    libDir = fileparts(libvlPath);
    fprintf('\nLibrary directory: %s\n', libDir);
else
    warning('libvl.dylib not found');
    libDir = '';
end

fprintf('\nCompiling...\n');

try
    % Build arguments manually
    args = {};
    args{end+1} = '-mexmaca64';
    args{end+1} = '-largeArrayDims';
    args{end+1} = '-O';
    args{end+1} = ['-I' vlfeatRoot];
    args{end+1} = ['-I' toolboxDir];
    args{end+1} = ['-I' fullfile(vlfeatRoot, 'vl')];
    args{end+1} = srcFile;
    args{end+1} = '-outdir';
    args{end+1} = mexOutDir;
    
    if ~isempty(libDir)
        args{end+1} = ['-L' libDir];
        args{end+1} = '-lvl';
    end
    
    fprintf('MEX command arguments:\n');
    for i = 1:length(args)
        fprintf('  %d: %s\n', i, args{i});
    end
    fprintf('\n');
    
    mex(args{:});
    
    mexFile = fullfile(mexOutDir, 'vl_sift.mexmaca64');
    if exist(mexFile, 'file')
        fprintf('\n✓ Success! MEX file created: %s\n', mexFile);
    else
        fprintf('\n✗ MEX file not found after compilation\n');
    end
    
catch ME
    fprintf('\n✗ Error: %s\n', ME.message);
    fprintf('Error location: %s\n', ME.stack(1).name);
    if length(ME.stack) > 1
        fprintf('Line: %d\n', ME.stack(1).line);
    end
end

