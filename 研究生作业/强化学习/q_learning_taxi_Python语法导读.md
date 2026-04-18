# `q_learning_taxi.py` 涉及的 Python 语法导读（面向小白）

本文只讲**语法与语言现象**，不讲强化学习公式。下面按**主题分类**组织，方便你按需查阅；同一知识点只在最贴切的一类里展开，其它类用「见 ××」互指。

---

## 阅读地图（分类目录）

| 大类 | 内容提要 |
|------|------------|
| **一、程序骨架与执行方式** | 模块/函数文档字符串、`if __name__ == "__main__"` |
| **二、模块、导入与库初始化** | `import`、`from … import`、`__future__`、matplotlib 先设后端 |
| **三、函数：定义、参数与调用** | `def`、`return`、位置/关键字参数、默认参数 |
| **四、类型注解与 typing** | 参数/返回值上的 `:类型`、`->` 、`List` / `Tuple` |
| **五、控制流** | `if` / `elif` / `else`、缩进、`for` / `while`、`range`、布尔运算 |
| **六、赋值、运算符与内置函数** | 解包、`_` 占位、增强赋值、`max` / `min` / `float` / `int` / `len` |
| **七、列表与序列用法** | `[]`、`.append`、切片（与 NumPy 切片对照） |
| **八、字符串与格式化** | f-string、相邻字符串自动拼接 |
| **九、文件路径** | `pathlib.Path`、`/`、`mkdir`、`__file__` |
| **十、命令行参数** | `argparse` |
| **十一、本脚本中的 NumPy** | 数组创建、索引、`argmax` 等 |
| **十二、随机数** | `random.Random(seed)` |
| **附录** | 按分类整理的速查表 |

---

## 一、程序骨架与执行方式

### 1.1 模块文档字符串（module docstring）

文件最上面用**三引号**包了一大段文字。这叫**模块级文档字符串**：给人和工具看的说明，**不参与运算逻辑**。

```1:15:研究生作业/强化学习/q_learning_taxi.py
"""
Taxi-v3 上的表格 Q-learning（Gymnasium 离散环境，等价于“格子世界”类任务）。
...
"""
```

函数里用三引号写的段落是**函数 docstring**，作用类似。

### 1.2 程序入口：`if __name__ == "__main__":`

```252:253:研究生作业/强化学习/q_learning_taxi.py
if __name__ == "__main__":
    main()
```

- **直接运行**本文件时，Python 把 **`__name__`** 设为 **`"__main__"`**，于是执行 **`main()`**。
- **被别的文件 `import`** 时，`__name__` 是模块名，**不会自动跑训练**，避免「一 import 就执行一整段副作用」。

---

## 二、模块、导入与库初始化

### 2.1 `from __future__ import annotations`

```17:17:研究生作业/强化学习/q_learning_taxi.py
from __future__ import annotations
```

**未来特性**，须放在文件**靠前**位置。这里配合**类型注解**：让注解可以**延后解析**，避免某些循环引用问题。小白可先记：**写类型提示时常见的一句**，不改变运行结果。

### 2.2 `import` 与 `from … import …`

- **`import 模块名`**：通过 **`模块名.成员`** 使用，如 `random.Random`、`plt.subplots`。
- **`from 模块 import 成员`**：把成员直接放进当前名字空间，如 `from pathlib import Path` 后可直接写 `Path`。

习惯上：**标准库** → **第三方库**，分块写，便于阅读（非语法强制）。

### 2.3 先设后端再 `import pyplot`

```26:29:研究生作业/强化学习/q_learning_taxi.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

`matplotlib` 的**后端**决定如何出图；无界面环境常用 **`"Agg"`**。须在 **`import matplotlib.pyplot`** 之前调用 **`use`**，否则可能已锁定后端。

---

## 三、函数：定义、参数与调用

### 3.1 `def`、`return`、docstring

```46:52:研究生作业/强化学习/q_learning_taxi.py
def epsilon_greedy(
    q_row: np.ndarray, n_actions: int, epsilon: float, rng: random.Random
) -> int:
    """ε-greedy：以概率 epsilon 随机探索，否则在当前 Q 行里贪心选最大值对应动作。"""
    if rng.random() < epsilon:
        return rng.randrange(n_actions)
    return int(np.argmax(q_row))
```

- **`def 名字(参数…):`** 定义函数；**`return`** 把值传回调用处，并结束当前这次调用。
- 参数后的 **`: 类型`**、行尾的 **`-> 类型`** 见**第四节**。

### 3.2 位置参数与关键字参数

```208:217:研究生作业/强化学习/q_learning_taxi.py
    q, returns = q_learning_train(
        train_env,
        n_episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
```

- **位置参数**：按定义顺序传，如第一个 `train_env`。
- **关键字参数**：**`形参名=实参`**，可读性好，顺序一般可与定义不一致（除非与位置参数混用时有规则限制）。

### 3.3 默认参数

```141:147:研究生作业/强化学习/q_learning_taxi.py
def save_result_plots(
    episode_returns: List[float],
    plot_dir: Path,
    mean_eval_return: float,
    mean_eval_len: float,
    window: int = 100,
) -> Path:
```

**`window: int = 100`**：调用时可省略 `window`，此时用 **100**。

---

## 四、类型注解与 typing（ing）

- **参数**：`q_row: np.ndarray`、`n_actions: int` 等，表示「约定类型」，**运行时 Python 默认不强制检查**。
- **返回值**：`**-> int**`、`**-> Path**`、`**-> None**` 等。
- **`from typing import List, Tuple`**：
  - **`List[float]`**：元素为 float 的列表（注解用）。
  - **`Tuple[np.ndarray, list]`**：二元组类型注解。

Python 3.9+ 也可写内置泛型如 **`list[float]`**；本文件用 **`List`** 与 **`from __future__ import annotations`** 搭配很常见。

---

## 五、控制流

### 5.1 分支：`if` / `elif` / `else` 与缩进

```36:41:研究生作业/强化学习/q_learning_taxi.py
    if system == "Darwin":
        candidates = ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]
    elif system == "Windows":
        candidates = ["Microsoft YaHei", "SimHei"]
    else:
        candidates = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Droid Sans Fallback"]
```

Python 用**缩进**（通常 4 空格）表示块，**没有** C 系语言的 **`{}`**。

### 5.2 循环：`for`、`while`、`range`

```79:85:研究生作业/强化学习/q_learning_taxi.py
    for ep in range(n_episodes):
        ...
        while not done:
```

- **`for 变量 in 可迭代对象:`**：`range(n)` 产生 `0 … n-1`。
- **`while 条件:`**：条件为真则反复执行循环体。

### 5.3 布尔运算：`not`、`or`

```82:89:研究生作业/强化学习/q_learning_taxi.py
        done = False
        ...
        while not done:
            ...
            done = terminated or truncated
```

- **`not done`**：对布尔值取反。
- **`a or b`**：至少一个为真则整体为真（本脚本里用于「终止或截断」）。

---

## 六、赋值、运算符与内置函数

### 6.1 解包与占位符 `_`

```81:88:研究生作业/强化学习/q_learning_taxi.py
        state, _ = env.reset(seed=seed + ep)
        ...
            next_state, reward, terminated, truncated, _ = env.step(action)
```

**`a, b, c = 可迭代对象`**：按位置拆成多个变量。**`_`** 表示「有位但不用」，避免多余变量名。

### 6.2 增强赋值

```97:99:研究生作业/强化学习/q_learning_taxi.py
            q[state, action] += alpha * (target - q[state, action])
            ep_return += reward
```

**`x += y`** 在多数情况下等价于 **`x = x + y`**（可变对象有细微差别，小白可先当等价）。

### 6.3 常用内置函数

- **`max(a, b)`**、**`min(...)`**：比较大小；本脚本用 **`max`** 限制探索率下限。
- **`float(x)`**、**`int(x)`**：类型转换。
- **`len(序列)`**：长度。

---

## 七、列表与序列用法

- **空列表**：`returns = []`；**字面量**：`candidates = ["PingFang SC", ...]`。
- **`.append(x)`**：末尾追加一项。
- **切片**（Python 列表与 NumPy 数组都常见）：
  - **`returns[-tail:]`**：从倒数第 `tail` 个元素一直到末尾。
  - 与 **`np.cumsum`** 等配合的切片见**第十一节**。

---

## 八、字符串与格式化

### 8.1 f-string

```225:227:研究生作业/强化学习/q_learning_taxi.py
    print(f"训练回合: {args.episodes}")
    print(f"评估 ({args.eval_episodes} 局, 贪心策略): 平均回报 = {mean_ret:.2f}, 平均步数 = {mean_len:.1f}")
```

- 前缀 **`f`** 表示 **f-string**，**`{表达式}`** 求值后插入字符串。
- **`{mean_ret:.2f}`**：格式化为保留两位小数的浮点数；**`.1f`** 为一位小数。

### 8.2 相邻字符串自动拼接

```163:167:研究生作业/强化学习/q_learning_taxi.py
    summary = (
        f"贪心评估（独立回合）\n"
        f"平均回报: {mean_eval_return:.2f}\n"
        f"平均步数: {mean_eval_len:.1f}"
    )
```

多个**字符串字面量**紧挨着，Python 会**自动拼成一段**（常配合括号折行）。

---

## 九、文件路径

```178:178:研究生作业/强化学习/q_learning_taxi.py
    path = plot_dir / "q_learning_taxi_results.png"
```

- **`Path`**（来自 `pathlib`）表示路径；**`/`** 在 `Path` 上表示拼子路径。
- **`plot_dir.mkdir(parents=True, exist_ok=True)`**：创建目录（含父目录），已存在不报错。

```199:199:研究生作业/强化学习/q_learning_taxi.py
        default=Path(__file__).resolve().parent / "results",
```

- **`__file__`**：当前源文件路径。
- **`.resolve()`**：规范为绝对路径。
- **`.parent`**：所在目录。合起来表示**默认输出到脚本同目录下的 `results/`**。

---

## 十、命令行参数：`argparse`

```186:204:研究生作业/强化学习/q_learning_taxi.py
    parser = argparse.ArgumentParser(description="Q-learning on Taxi-v3 (Gymnasium)")
    parser.add_argument("--episodes", type=int, default=20000, help="训练回合数")
    ...
    parser.add_argument("--render-eval", action="store_true", help="评估时用 human 渲染一局")
    ...
    args = parser.parse_args()
```

- **`ArgumentParser`**：建立解析器；**`add_argument`** 定义一个选项。
- **`type=int`** / **`type=float`** / **`type=Path`**：把命令行字符串转成对应类型。
- **`default=…`**：未传入时的默认值。
- **`action="store_true"`**：**不需要值**；出现 `--render-eval` 则对应属性为 **`True`**。
- **`parse_args()`** 得到 **`args`**；**`--eval-episodes`** 对应 **`args.eval_episodes`**（横杠变下划线）。

---

## 十一、本脚本中的 NumPy

前提：**`import numpy as np`**，API 多写成 **`np.名字`**。

| 用法 | 含义（结合本脚本） |
|------|---------------------|
| **`np.zeros((行, 列), dtype=np.float64)`** | 全零二维表，即 Q 表初值 |
| **`q[state, action]`** | 一行一列一个格子 |
| **`q[state]`** | 取一整行（一维数组） |
| **`np.argmax(q_row)`** | 最大值的**下标**（第几个动作） |
| **`np.max(数组)`** | 最大值 |
| **`np.arange(1, n+1)`** | 横轴回合编号 |
| **`np.asarray(...)`** | 转成 ndarray |
| **`np.mean(...)`** | 平均；常与 **`returns[-tail:]`** 联用 |
| **`np.nan`** | 非数，作占位 |
| **`np.full`、`np.cumsum`、`np.insert`** | 在 **`moving_average`** 里做滑动平均 |
| **`arr.astype(np.float64)`**、**`.copy()`** | 类型转换与拷贝 |

---

## 十二、随机数：`random.Random`

```71:71:研究生作业/强化学习/q_learning_taxi.py
    rng = random.Random(seed)
```

创建**独立随机源**，给定 **`seed`** 可复现实验。后续 **`rng.random()`**、**`rng.randrange(n_actions)`** 都走 **`rng`**，比直接用模块级 **`random.*`** 更易控制。

---

## 附录：按分类速查

| 大类 | 关键词 |
|------|--------|
| 程序骨架 | `"""…"""`，`if __name__ == "__main__"` |
| 导入与初始化 | `import`，`from … import`，`from __future__`，`matplotlib.use("Agg")` |
| 函数 | `def`，`return`，默认参数，位置/关键字调用 |
| 类型 | `:类型`，`->类型`，`List`，`Tuple` |
| 控制流 | `if/elif/else`，`for`，`while`，`range`，`not`，`or` |
| 赋值与内置 | 解包，`_`，`+=`，`max`，`float`，`int`，`len` |
| 列表与切片 | `[]`，`append`，`[-tail:]` 等 |
| 字符串 | `f"…"`，`{x:.2f}` |
| 路径 | `Path`，`/`，`mkdir`，`__file__` |
| 命令行 | `argparse`，`type`，`default`，`action="store_true"` |
| NumPy | `zeros`，`argmax`，索引，`mean`，`nan` 等 |
| 随机 | `random.Random(seed)` |

若你需要把某一类（例如只 **`argparse`** 或只 **`moving_average` 里的切片与 cumsum`**）扩成「逐行注释版」，可以指定函数名或行号区间再拆一篇。
