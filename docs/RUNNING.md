# 本地运行与验证

只查看项目方案与结果，可阅读 [README](../README.md) 和 [Case Study](CASE_STUDY.md)，无需安装环境。下面步骤用于在自己的电脑运行 Demo；GitHub 仓库本身不是在线聊天页面。

## 1. 获取代码并安装依赖

以下命令适用于 Windows PowerShell。解压 HR 用 ZIP 后也可直接进入包含 `app.py` 的目录，跳过克隆步骤。

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

完整客服回答需要可访问 DeepSeek API 的网络、有效密钥和可用额度，会产生 API 费用。模型名可通过环境变量切换，但必须是账户实际可用的模型；变更模型后的运行不能视为原模型基线的复现。纯建库、检索和离线单元检查不调用 DeepSeek。

## 3. 首次建立本地知识索引

仓库已包含六份 `knowledge_base/*.md` 原文与建库脚本，不需要获取作者电脑上的数据库。模型缓存和向量库属于本地运行产物：

| 目录 | 如何得到 |
|---|---|
| `.model_cache/` | 程序首次加载 `BAAI/bge-small-zh-v1.5` 时从 Hugging Face 下载，后续复用 |
| `chroma_db/` | 由本项目的知识原文、切分代码和 BGE 向量生成 |

**没有数据库的新环境**执行一次：

```powershell
.\.venv\Scripts\python.exe scripts/build_vector_store.py
```

首次下载需要能够访问 `huggingface.co`，耗时取决于网络。建库不需要 DeepSeek 密钥，不会调用 DeepSeek。使用当前六份原文，正常结束应显示：

```text
Collection：aura_customer_service
知识片段数量：122
Chroma 实际记录：122
```

已建立可用数据库的环境跳过本步。日常启动不会重新切分或建库；脚本发现已有非空 Collection 会停止，避免覆盖。`--rebuild` 会删除并重建该 Collection，仅限明确需要修复自己的索引时使用，不是普通启动步骤。

## 4. 检查检索并启动界面

先验证本地知识链路，不调用生成 API：

```powershell
.\.venv\Scripts\python.exe scripts/test_retrieval.py
```

输入“洁面多少钱？”，应看到最多五条实际片段，以及来源、标题、编号和距离。输入 `exit` 或 `quit` 退出。

启动网页：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

打开终端显示的地址，通常是 `http://localhost:8501`。按顺序测试“洁面多少钱？”“订单 123456 到哪里了？”“123456 退款到账了吗？”“我要人工客服。”，展开执行详情查看 RAG、模拟工具和 HANDOFF。

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

上述验证没有重新执行全新环境安装、首次联网下载或 DeepSeek 端到端调用，不保证任意环境下安装一次即可成功，也不更新历史 Eval 成绩。原始基线及其口径见 [项目状态](../PROJECT_STATUS.md)。

离线单元检查：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

该命令不调用 DeepSeek、不加载 BGE、不连接 Chroma。批量 RAG / Agent Eval 则会产生 API 调用并写入结果，日常演示无需运行。
