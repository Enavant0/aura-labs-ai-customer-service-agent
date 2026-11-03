# 本地运行与验证

本页说明怎样在本地运行项目。只想了解能力与结果，可先读 [README](../README.md) 和 [项目案例](CASE_STUDY.md)；GitHub 仓库本身不提供在线聊天。

运行需要两部分：本地知识索引，以及调用 DeepSeek 的有效配置。建库和纯检索可以先独立验证，不需要调用生成 API。

## 1. 获取代码并安装依赖

命令适用于 Windows PowerShell。使用 ZIP 时，解压后进入包含 `app.py` 的目录，跳过克隆。

```powershell
git clone https://github.com/Enavant0/aura-labs-ai-customer-service-agent.git aura-rag-customer-service
cd .\aura-rag-customer-service

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

已有可用虚拟环境时，跳过重复创建和安装。已验证环境为 Windows、Python 3.14.8；BGE 使用 CPU，不要求 GPU。

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

完整问答需要有效密钥、账户可用模型、额度和网络，会产生 API 费用。这里的 `deepseek-flash` 是历史评测使用的配置；如果切换模型，应单独记录，不能当作相同条件下复现原成绩。建库、纯检索和离线单元检查不调用 DeepSeek。

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

已有可用知识库时跳过本步，日常启动不重建。Collection 指 Chroma 中存放知识记录的集合；脚本发现集合非空会停止，避免覆盖已有记录。`--rebuild` 会删除并重建该集合，只在明确需要修复索引时使用。

## 4. 检查检索并启动界面

先验证本地知识链路，不调用生成 API：

```powershell
.\.venv\Scripts\python.exe scripts/test_retrieval.py
```

输入“洁面多少钱？”，应显示最多五个知识片段及其来源、标题、编号和距离。当前使用余弦距离，数值越小表示向量方向越接近、排名越靠前。输入 `exit` / `quit` 退出。

启动网页：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

打开终端地址（通常为 `http://localhost:8501`），按 [README 四步演示](../README.md) 提问并展开执行详情。

也可使用同一服务的命令行入口：

```powershell
.\.venv\Scripts\python.exe scripts/test_agent.py
```

订单和退款读取固定的本地模拟数据。HANDOFF 表示人工处理或补充信息提示，没有实际接通客服。每题独立，页面显示历史不作为 Agent 记忆。

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

已在临时项目副本中，使用既有依赖和 BGE 缓存验证了首次建库与前五个结果检索：生成 122 条记录，产品查询返回五个结果并命中产品来源，完整 88 项离线测试通过。原演示数据库未改动。

验证所用依赖如下，供环境排查参考；`requirements.txt` 尚未锁定完整依赖树，以下表格不是锁文件：

| 依赖 | 验证版本 |
|---|---|
| openai | 3.35.1 |
| python-dotenv | 1.2.4 |
| streamlit | 1.66.0 |
| chromadb | 1.5.9 |
| sentence-transformers | 6.1.0 |
| torch | 2.14.1 |

这项验证没有重新覆盖全新依赖安装、首次联网下载或完整 DeepSeek 问答调用，因此不能保证所有新环境一次运行成功。历史评测成绩没有改变，范围和口径见 [项目状态](../PROJECT_STATUS.md)。

离线单元检查：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

该命令不调用 DeepSeek、不加载 BGE、不连接 Chroma。批量 RAG / Agent Eval 则会产生 API 调用并写入结果，日常演示无需运行。
