# VLFeat 架构兼容性问题解决方案

## 问题说明

如果您看到以下错误：
```
函数或变量 'vl_sift' 无法识别
MATLAB is running as ARM64 (mexmaca64) but MEX files are x86_64 (mexmaci64)
```

这是因为：
- MATLAB 正在以 ARM64（Apple Silicon 原生）模式运行
- 但 VLFeat 的 MEX 文件是 x86_64 架构
- ARM64 MATLAB 无法加载 x86_64 MEX 文件

## 解决方案

### 方案1：通过 Rosetta 2 运行 MATLAB（推荐）

**方法A：使用终端启动**
1. 退出当前 MATLAB
2. 打开终端（Terminal）
3. 运行以下命令（请根据您的 MATLAB 版本调整路径）：
   ```bash
   arch -x86_64 /Applications/MATLAB_R2024a.app/bin/matlab
   ```
   或者查找您的 MATLAB 安装路径：
   ```bash
   which matlab
   # 然后使用：
   arch -x86_64 [MATLAB路径]
   ```

**方法B：在 Finder 中设置**
1. 退出 MATLAB
2. 在 Finder 中找到 MATLAB 应用程序
3. 右键点击 MATLAB，选择"显示简介"（Get Info）
4. 勾选"使用 Rosetta 打开"（Open using Rosetta）
5. 重新启动 MATLAB

### 方案2：检查当前 MATLAB 架构

在 MATLAB 命令窗口中运行：
```matlab
mexext
computer('arch')
```

- 如果显示 `mexmaca64` 或 `maca64`：MATLAB 是 ARM64 版本
- 如果显示 `mexmaci64` 或 `maci64`：MATLAB 是 x86_64 版本（可以通过 Rosetta 2 运行）

## 验证

启动 MATLAB 后，运行：
```matlab
mexext
```

如果显示 `mexmaci64`，说明 MATLAB 正在 x86_64 模式下运行，可以正常加载 VLFeat MEX 文件。

然后运行 `SFMedu2.m` 应该可以正常工作。

