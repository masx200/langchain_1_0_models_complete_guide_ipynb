# LangChain 1.0 模型声明与使用完全指南

本项目提供了LangChain 1.0中模型声明和使用的全面指南，详细介绍了三种主要的模型配置模式及其灵活应用。

## 📋 目录

- [概述](#概述)
- [功能特性](#功能特性)
- [安装要求](#安装要求)
- [快速开始](#快速开始)
- [三种主要模式](#三种主要模式)
- [使用场景](#使用场景)
- [最佳实践](#最佳实践)
- [许可证](#许可证)

## 概述

LangChain 1.0引入了一种统一的模型接口设计，支持15+个AI模型提供商（OpenAI、Anthropic、Google等）。本指南深入讲解了模型的声明方式、配置参数和实际应用，帮助开发者灵活高效地使用各种大语言模型。

## 功能特性

- 🚀 **统一接口**：一套代码支持多个模型提供商
- 🔧 **灵活配置**：支持运行时动态切换模型和参数
- 📚 **三种模式**：固定模型、完全可配置、可配置+默认值
- 🎯 **智能推断**：自动识别模型提供商
- 🛠️ **工具绑定**：支持函数调用和工具集成
- 📊 **参数控制**：temperature、max_tokens等完整参数支持

## 安装要求

### Python版本
Python 3.8+

### 依赖包
```bash
pip install langchain langchain-openai langchain-anthropic langchain-google-vertexai
```

### API密钥配置
根据使用的模型提供商配置相应的API密钥：

```python
import os
from google.colab import userdata

# OpenAI
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# Anthropic (可选)
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

# Google Vertex AI (可选)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/credentials.json"
```

## 快速开始

### 最简单的使用方式
```python
from langchain.chat_models import init_chat_model

# 声明并使用模型
model = init_chat_model("openai:gpt-4o", temperature=0)
response = model.invoke("你好，请介绍一下你自己")
print(response.content)
```

### 自动推断提供商
```python
# 不需要显式指定提供商，系统自动识别
model = init_chat_model("gpt-4o-mini", temperature=0)
response = model.invoke("讲一个笑话")
```

## 三种主要模式

### 1. 固定模型 (Fixed Model)
**特点**：模型和参数在初始化时固定

```python
# 最直接的使用方式
model = init_chat_model("openai:gpt-4o", temperature=0, max_tokens=100)
response = model.invoke("what's your name")
```

**适用场景**：
- 单一模型应用
- 生产环境部署
- 明确的模型需求

### 2. 完全可配置模型 (Fully Configurable)
**特点**：不指定默认模型，运行时动态指定

```python
# 创建可配置模型
configurable_model = init_chat_model(temperature=0)

# 运行时指定模型
response = configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "gpt-4o"}}
)
```

**适用场景**：
- 多模型切换
- A/B测试
- 用户选择模型
- 成本优化策略

### 3. 可配置+默认值模型 (Configurable with Default)
**特点**：有默认值，可选择性覆盖

```python
# 带默认值的可配置模型
model = init_chat_model(
    "openai:gpt-4o",
    configurable_fields="any",
    temperature=0
)

# 使用默认配置
response1 = model.invoke("what's your name")

# 运行时覆盖配置
response2 = model.invoke(
    "what's your name",
    config={"configurable": {"model": "claude-sonnet", "temperature": 0.6}}
)
```

**适用场景**：
- 大部分场景用默认配置
- 特殊情况需要切换模型
- 需要灵活性的生产环境

## 高级特性

### 配置前缀 (config_prefix)
使用命名空间管理多个可配置模型：

```python
# 总结器模型
summarizer = init_chat_model(
    "gpt-4o-mini",
    configurable_fields="any",
    config_prefix="summarizer",
    temperature=0
)

# 翻译器模型
translator = init_chat_model(
    "gpt-4o-mini",
    configurable_fields="any",
    config_prefix="translator",
    temperature=0
)

# 统一配置
unified_config = {
    "configurable": {
        "summarizer_model": "gpt-4o-mini",
        "translator_model": "gpt-4o",
    }
}

summary = summarizer.invoke("总结这段文本", config=unified_config)
translation = translator.invoke("翻译这段文本", config=unified_config)
```

### 运行时参数覆盖
```python
# 运行时同时指定模型和参数
response = configurable_model.invoke(
    "讲一个笑话",
    config={
        "configurable": {
            "model": "gpt-4o-mini",
            "temperature": 0.9,
            "max_tokens": 100
        }
    }
)
```

### 智能模型选择
```python
def ask_model(question: str, use_advanced: bool = False):
    """根据需求选择模型"""
    model_name = "gpt-4o" if use_advanced else "gpt-4o-mini"
    
    flexible_model = init_chat_model(temperature=0)
    
    response = flexible_model.invoke(
        question,
        config={"configurable": {"model": model_name}}
    )
    return response.content

# 简单问题用小模型
simple_answer = ask_model("1+1=?", use_advanced=False)

# 复杂问题用大模型
complex_answer = ask_model("解释量子纠缠", use_advanced=True)
```

## 使用场景

### 1. 成本优化策略
```python
# 默认使用便宜的小模型
default_model = init_chat_model(
    "gpt-4o-mini",
    configurable_fields="any",
    temperature=0
)

# 复杂任务时切换到更强模型
def process_request(question, complexity="simple"):
    model = "gpt-4o" if complexity == "complex" else "gpt-4o-mini"
    
    return default_model.invoke(
        question,
        config={"configurable": {"model": model}}
    )
```

### 2. A/B测试
```python
# 创建两个配置的模型用于A/B测试
model_a = init_chat_model("gpt-4o-mini", temperature=0.7)
model_b = init_chat_model("gpt-4o-mini", temperature=0.9)

# 分配用户到不同组进行测试
def ab_test_response(question, user_group):
    model = model_a if user_group == "A" else model_b
    return model.invoke(question)
```

### 3. 多功能应用系统
```python
# 创建一个统一的多功能模型接口
multi_model = init_chat_model(
    "gpt-4o-mini",
    configurable_fields="any",
    temperature=0
)

def chat_system(question, mode="general"):
    configs = {
        "coding": {"model": "gpt-4o", "temperature": 0.2},
        "creative": {"model": "gpt-4o", "temperature": 0.8},
        "general": {"model": "gpt-4o-mini", "temperature": 0.5}
    }
    
    return multi_model.invoke(
        question,
        config={"configurable": configs[mode]}
    )
```

## 最佳实践

### 1. 开发阶段
- 使用可配置模型快速实验不同模型
- 利用 `temperature` 参数测试不同输出风格
- 使用小模型进行初步测试，节省成本

### 2. 生产环境
- **稳定性优先**：使用固定模型或设置明确的默认模型
- **监控切换**：记录模型调用情况，便于追踪问题
- **错误处理**：为模型切换设置fallback机制

### 3. 多模型应用
- 使用 `config_prefix` 管理多个模型的配置
- 统一配置管理，避免配置分散
- 建立模型选择的策略和规则

### 4. 成本优化
- 默认使用性能-成本比高的小模型
- 复杂任务时切换到大模型
- 设置合理的 `max_tokens` 限制输出长度
- 考虑使用批处理减少API调用次数

### 5. 工具调用
- 选择支持工具调用的模型（GPT-4o, Claude等）
- 使用 `.bind_tools()` 为模型添加函数调用能力
- 合理设计工具接口，减少调用复杂度

### 6. 参数配置指南

| 参数 | 推荐值 | 场景 |
|------|--------|------|
| temperature | 0 | 确定性输出，代码生成，事实问答 |
| temperature | 0.3-0.5 | 一般对话，创意任务 |
| temperature | 0.8-1.0 | 创意写作，头脑风暴 |
| max_tokens | 100 | 简短回答 |
| max_tokens | 500 | 详细解释 |
| max_tokens | 1000+ | 长文本生成 |

## 支持的模型提供商

- **OpenAI**: GPT-4, GPT-4o, GPT-3.5
- **Anthropic**: Claude Sonnet, Claude Haiku
- **Google**: Gemini, Vertex AI
- **其他**: 支持15+主流模型提供商

## 许可证

本项目遵循 MIT 许可证。详情请查看 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个指南！

## 作者

**MiniMax Agent** - LangChain 1.0 模型使用指南

---

*本指南基于LangChain 1.0版本编写，建议定期更新以跟随最新版本。*