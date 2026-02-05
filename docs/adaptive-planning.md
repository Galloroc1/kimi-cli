# Adaptive Planning (动态 DAG 规划)

自适应规划是 Kimi CLI 的一种高级执行模式，它使用层次化规划来执行复杂任务。

## 工作原理

```
用户输入
    │
    ▼
┌─────────────────┐
│  Planner        │  LLM 生成高层阶段图
│  (阶段级规划)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  StageExecutor  │  每个阶段使用 ReAct 执行
│  (步骤级执行)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Controller     │  根据结果决定转换/重规划
│  (协调控制)      │
└─────────────────┘
```

## 启用方法

### 方法 1: 配置文件

编辑 `~/.config/kimi-cli/config.toml`:

```toml
[loop_control]
adaptive_planning = true
max_steps_per_turn = 100
```

### 方法 2: 环境变量

```bash
export KIMI_LOOP_CONTROL__ADAPTIVE_PLANNING=true
kimi
```

## 对比

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| 默认模式 | 单轮 ReAct | 简单对话 |
| Ralph 模式 | 预定义循环 | 自动化迭代 |
| **Adaptive (新)** | **动态 DAG** | **复杂多阶段任务** |

## 示例

启用自适应规划后，执行复杂任务：

```
$ kimi
> 分析这个 Python 项目的性能问题并优化

[Planner] 生成执行计划:
  1. 分析项目结构
  2. 识别性能瓶颈
  3. 实施优化
  4. 验证结果

[Stage: 分析项目结构]
  - 读取项目文件...
  - 分析依赖关系...
  ✓ 阶段完成

[Stage: 识别性能瓶颈]
  - 运行性能分析...
  - 发现热点函数...
  ✓ 阶段完成

...
```

## 配置选项

```toml
[loop_control]
# 启用自适应规划
adaptive_planning = true

# 每轮最大步数（阶段内步数总和）
max_steps_per_turn = 100

# 单个阶段最大重试次数
max_retries_per_step = 3

# 最大重规划次数（遇到失败时）
# 注意：这个值在代码中硬编码为 3，如需修改请改源码
```

## 测试

### 运行单元测试

```bash
cd /Users/ran/pyproject/kimi-cli
pytest tests/core/test_adaptive.py -v
```

### 运行手动测试

```bash
cd /Users/ran/pyproject/kimi-cli
python test_adaptive_manual.py
```

## 实现细节

### 核心组件

- `Planner`: 生成和调整执行计划
- `StageExecutor`: 使用 ReAct 执行单个阶段
- `ExecutionController`: 协调整个执行流程

### 阶段类型

每个阶段包含：
- `id`: 唯一标识
- `name`: 阶段名称
- `goal`: 阶段目标
- `exit_criteria`: 完成标准
- `max_iterations`: 最大迭代次数
- `tools_hint`: 建议使用的工具

### 转换类型

阶段间转换可以是：
- `next`: 正常进入下一阶段
- `retry`: 重试当前阶段
- `backtrack`: 回退到之前阶段
- `replan`: 重新生成整个计划
- `finish`: 任务完成

## 调试

启用调试日志：

```bash
export KIMI_LOG_LEVEL=DEBUG
kimi
```

查看生成的计划：
- 计划生成后会显示在 UI 中
- 每个阶段开始时会显示阶段信息
- 转换时会显示转换类型

## 限制

1. 需要 Python 3.12+
2. 计划生成需要额外的 LLM 调用
3. 复杂任务可能需要更多 token
4. 目前不支持并行阶段执行

## 未来改进

- [ ] 支持并行阶段执行
- [ ] 可视化执行图
- [ ] 手动干预和修改计划
- [ ] 学习历史执行优化计划生成
