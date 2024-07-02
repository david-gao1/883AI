## 通过conda创建python执行环境
-- 尽量别使用base环境，因为其他环境会依赖base环境，而不同的环境对于同一个包依赖的版本可能不同，导致冲突。


```bash

# 创建虚拟环境
conda create --name 883ai python=3.8
conda activate 883ai

# 看是否pip安装在此虚拟环境中
whereis pip
# pip: /Users/lianggao/miniforge3/envs/883ai/bin/pip


# 查看已经安装依赖
 conda list 

# 导出依赖
pip freeze > requirements.txt

```