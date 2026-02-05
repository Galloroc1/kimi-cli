---
name: code-review-flow
description: A structured code review workflow that guides the agent through systematic code analysis, issue identification, and feedback generation. Use when the user wants to perform a thorough code review with multiple stages of analysis.
type: flow
---

# Code Review Flow

This skill provides a structured workflow for performing code reviews.

## Process

```mermaid
flowchart TD
    BEGIN[Begin] --> overview[获取代码概览]
    overview --> analyze[分析代码结构]
    analyze --> issues[识别潜在问题]
    issues --> decision{发现问题?}
    decision --是--> feedback[生成反馈建议]
    decision --否--> summary[生成总结]
    feedback --> summary
    summary --> END[End]
```

## Stages

1. **获取代码概览** - 理解代码的整体结构和目的
2. **分析代码结构** - 检查代码组织、命名规范、设计模式
3. **识别潜在问题** - 查找bug、性能问题、安全隐患
4. **生成反馈建议** - 提供具体的改进建议（如果发现问题）
5. **生成总结** - 输出完整的代码审查报告

## Decision Points

- `发现问题?` - 如果识别到任何问题，进入反馈生成阶段；否则直接生成总结
