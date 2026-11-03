# 本地运行与验证

本页用于本地复现；仅查看方案与结果请读 [README](../README.md) / [Case Study](CASE_STUDY.md)。GitHub 不是在线聊天页面。

## 1. 获取代码并安装依赖

命令适用于 Windows PowerShell。使用 ZIP 时，解压后进入包含 `app.py` 的目录，跳过克隆。

```powershell
git clone https://github.com/Enavant0/aura-labs-ai-customer-service-agent.git aura-rag-customer-service
cd .\aura-rag-customer-service

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

已有可用虚拟环境时无需重复创建或安装。验证环境为 Windows、Python 3.14.8，BGE 使用 CPU，不要求 GPU。

## 2. 配置 DeepSeek

仅在没有 `.env` 时复制模板，保留已有配置：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

在本地 `.env` 填写自己的密钥：

```dotenv
DEEPSEEK_API_KEY=填写自己的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

问答需要有效密钥、账户可用模型、额度及网络，会产生 API 费用；切换模型不视为复现原基线。建库、检索及离线单元检查不调用 DeepSeek。

## 3. 首次建立本地知识索引

六份 `knowledge_base/*.md` 原文和建库脚本已提供，不需复制作者的数据：

| 目录 | 如何得到 |
|---|---|
| `.model_cache/` | 程序首次加载 `BAAI/bge-small-zh-v1.5` 时从 Hugging Face 下载，后续复用 |
| `chroma_db/` | 由本项目的知识原文、切分代码和 BGE 向量生成 |

**没有数据库的新环境**执行一次：

```powershell
.\.venv\Scripts\python.exe scripts/build_vector_store.py
```

首次下载需访问 `huggingface.co`。建库不需要 DeepSeek 密钥，使用当前资料应显示：

```text
Collection：aura_customer_service
知识片段数量：122
Chroma 实际记录：122
```

已有可用库时跳过本步，日常启动不重建。脚本遇到非空 Collection 会停止；`--rebuild` 会删除并重建该集合，仅限明确修复索引时使用。

## 4. 检查检索并启动界面

先验证本地知识链路，不调用生成 API：

```powershell
.\.venv\Scripts\python.exe scripts/test_retrieval.py
```

输入“洁面多少钱？”，查看最多五条片段的来源、标题、编号和距离；`exit` / `quit` 退出。

启动网页：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

打开终端地址（通常为 `http://localhost:8501`），按 [README 四步演示](../README.md) 提问并展开执行详情。

也可使用同一服务的命令行入口：

```powershell
.\.venv\Scripts\python.exe scripts/test_agent.py
```

订单和退款均为本地固定模拟数据；HANDOFF 仅提示人工处理。每题独立，页面显示历史不作为 Agent 记忆。

## 5. 常见问题

| 提示或现象 | 处理方式 |
|---|---|
| 缺少依赖或导入失败 | 确认安装和启动使用同一个 `.venv\Scripts\python.exe` |
| BGE 加载或下载失败 | 检查到 Hugging Face 的网络、磁盘空间和目录写入权限；不要换模型来掩盖问题 |
| `chroma_db` 不存在 / Collection 不存在 | 新环境先执行第 3 步，网页不会自动建库 |
| Collection 为空或建库中断 | 先检查原错误；明确需要重建自己的残缺索引时才使用 `--rebuild` |
| 未配置密钥 / HTTP 401 | 检查项目根目录 `.env` 中的密钥，勿把密钥贴到 Issue 或提交 Git |
| HTTP 402 / 404 | 分别检查账户额度、接口与模型名称是否可用 |
| API 超时或限流 | 检查网络或稍后重试；程序只对指定临时错误有限重试 |
| 能打开页面但尚不能回答 | 确认密钥和知识索引均已准备；服务在首次提交问题时初始化 |

不要上传 `.env`、虚拟环境、模型缓存、数据库或真实运行日志；这些目录已有 Git 忽略规则。

## 6. 验证范围与版本

已在临时项目副本中，使用既有依赖和 BGE 缓存完成首次建库及 Top 5 检索验证：122 条记录，产品查询取回五条结果并命中产品来源，完整 88 项离线测试通过。原演示数据库未改动。

验证所用依赖如下，供环境排查参考；`requirements.txt` 尚未锁定完整依赖树，以下表格不是锁文件：

| 依赖 | 验证版本 |
|---|---|
| openai | 3.35.1 |
| python-dotenv | 1.2.4 |
| streamlit | 1.66.0 |
| chromadb | 1.5.9 |
| sentence-transformers | 6.1.0 |
| torch | 2.14.1 |

未复测全新安装、首次联网下载或 DeepSeek 端到端调用，不保证跨环境一次成功；历史 Eval 不变，口径见 [项目状态](../PROJECT_STATUS.md)。

离线单元检查：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

该命令不调用 DeepSeek、不加载 BGE、不连接 Chroma。批量 RAG / Agent Eval 则会产生 API 调用并写入结果，日常演示无需运行。
