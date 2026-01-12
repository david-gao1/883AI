# 快速解决方案

## 问题
MATLAB 正在以 ARM64 模式运行，但 VLFeat MEX 文件是 x86_64 架构，无法加载。

## 立即解决步骤

### 步骤 1：退出当前 MATLAB
关闭所有 MATLAB 窗口

### 步骤 2：找到 MATLAB 安装路径
在终端运行：
```bash
find /Applications -name "matlab" -type f | grep "MATLAB.*app/bin/matlab"
```

或者查看常见位置：
- `/Applications/MATLAB_R2024a.app/bin/matlab`
- `/Applications/MATLAB_R2023b.app/bin/matlab`
- `/Applications/MATLAB_R2023a.app/bin/matlab`
- `/Applications/MATLAB_R2022b.app/bin/matlab`

### 步骤 3：使用 Rosetta 2 启动 MATLAB

**方法A：使用提供的脚本**
```bash
cd /Users/lianggao/MyWorkSpace/002install/matlab/视觉/Assignment1/SFMedu2
./start_matlab_rosetta.sh
```

**方法B：手动启动（替换下面的路径为您的MATLAB路径）**
```bash
arch -x86_64 /Applications/MATLAB_R2024a.app/bin/matlab
```

**方法C：在 Finder 中设置（永久）**
1. Finder → 应用程序 → 找到 MATLAB
2. 右键点击 MATLAB → "显示简介"
3. 勾选 "使用 Rosetta 打开"
4. 关闭窗口，正常启动 MATLAB

### 步骤 4：验证
在 MATLAB 中运行：
```matlab
mexext
```
应该显示 `mexmaci64`（不是 `mexmaca64`）

### 步骤 5：运行代码
```matlab
cd('/Users/lianggao/MyWorkSpace/002install/matlab/视觉/Assignment1/SFMedu2')
SFMedu2
```

## 说明
- ARM64 MATLAB 无法加载 x86_64 MEX 文件
- 通过 Rosetta 2 运行 MATLAB 可以让它在 x86_64 模式下运行
- 这样就能加载 x86_64 的 MEX 文件了

