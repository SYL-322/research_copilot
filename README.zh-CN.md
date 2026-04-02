# Research Copilot

[English](./README.md) | 简体中文

`research_copilot` 是一个本地优先的研究助手，主要用于在命令行中处理学术论文。

它同时也是一个 vibe coding 项目：整体实现强调实用、可迭代、产品导向，而不是过度工程化。

它围绕一个实用工作流构建：

- 将论文导入本地存储
- 为论文构建结构化 memory，便于后续检索
- 围绕单篇论文进行有依据的问答
- 结合检索元数据和本地论文 memory 对 topic 进行扫描
- 持续维护 topic 的历史快照
- 生成近期论文的 daily digest

这个项目刻意保持轻量。它使用 Python、SQLite、`data/` 下的本地文件，以及直接的 OpenAI API 调用；不依赖 LangChain、向量数据库或大型编排框架。

## 目录

- [这个项目是什么](#what-this-project-is)
- [核心组件](#key-components)
- [工作原理](#how-it-works)
- [环境准备](#setup)
- [使用方式](#how-to-use)
  - [1. Ingest 一篇论文](#ingest-a-paper)
  - [2. 列出已存储论文](#list-stored-papers)
  - [3. 搜索本地论文](#search-stored-papers)
  - [4. 构建 paper memory](#build-paper-memory)
  - [5. 围绕单篇论文提问](#ask-questions-about-a-paper)
  - [6. 运行 topic scan](#run-topic-scan)
  - [7. 生成 daily digest](#generate-a-daily-digest)
- [Topic Report 输出](#topic-report-output)
- [设计理念](#design-philosophy)
- [局限性](#limitations)
- [成本考虑](#cost-considerations)
- [后续可改进方向](#future-improvements)
- [测试](#tests)

<a id="what-this-project-is"></a>
## 这个项目是什么

这个系统主要解决两个相关但不同的问题：

1. 充分理解单篇论文，从而提出更有价值的后续问题。
2. 持续维护一个研究方向的整体视图，同时仍然能追溯到具体论文。

项目把这两类任务分开处理：

- **论文工作流**：PDF/arXiv ingest -> 文本 chunks -> 结构化 paper memory -> grounded QA
- **主题工作流**：文献检索 -> 候选元数据 + 本地 memories -> 结构化 topic report -> 版本化 topic memory
- **摘要工作流**：近期论文检索 -> 元数据分诊 -> daily digest

更准确地说，它是一个本地研究工作台，而不是一个“全自动研究代理”。

它也刻意保持 vibe coding 的气质：优先小而清晰的本地工具、快速迭代、可检查的数据文件，以及直接可用的 CLI 工作流。

<a id="key-components"></a>
## 核心组件

### Paper memory

Paper memory 是针对单篇论文的结构化 JSON 摘要，由 LLM 基于 ingest 后的论文文本生成。当前 schema 包括：

- problem
- core idea
- method overview
- key components
- design rationale
- assumptions
- contributions
- limitations
- glossary
- failure modes
- open questions

目标是得到一种可复用、压缩但结构清晰的论文表示形式，它比普通摘要更结构化，也比只存原始 chunks 更实用。

如果长论文在送入 LLM 前被截断，存储的 memory 会带上 `truncated: true`，方便下游流程将其视为可能不完整。

### Grounded QA

Paper QA 回答单篇论文的问题时，会同时使用两类信息：

- 已存储的 paper memory
- 从论文原文中检索出的相关 chunks

Prompt 会要求模型优先使用检索到的文本片段作为证据，并在依赖这些证据时引用 `chunk_index`。这样做是为了尽量把回答绑定到本地证据，而不是依赖模型自由回忆。

如果一个问题没有检索到足够的支持 chunks，QA 会在 prompt 里明确标记这一点，并更保守地仅基于 paper memory 作答。

### Topic scanning

Topic scan 会基于以下来源，为某个研究 topic 生成结构化报告：

- arXiv 检索结果
- Semantic Scholar 检索结果
- 与 topic 有词汇重叠的本地 paper memories
- 之前 topic reports 的简短摘要

报告内容包括：子主题、代表论文、近期趋势、开放问题、证据质量说明、缺失方向、研究前沿总结、检索统计信息，以及完整的候选论文附录。

当前 topic scan 有两种 grounding 模式：

- **metadata-only mode**：只有检索元数据可用，因此报告会对方法细节和局限性保持保守
- **memory-backed mode**：候选集中的部分论文有本地 memory，因此报告可以做更深入的跨论文比较

### Topic memory

Topic memory 是历史 topic reports 的版本化记录。每次成功的 topic scan 都可以在 SQLite 中保存一个快照。后续扫描时，系统会读取最近快照的简短总结，并要求模型把这些历史结论视为“待验证假设”，而不是事实本身。

这是一个持续演化的 memory 层，不是真实世界的 ground truth 知识库。

### Two-pass topic scan

Topic scan 支持两阶段流程：

- 第 1 阶段使用 light model
- 第 2 阶段只在质量启发式、复杂度启发式命中，或显式使用 `--high-quality` 时执行

第二阶段会接收第一阶段的初始 JSON 报告，并尝试改进比较、缺口分析和前沿判断。

### Model routing

项目区分两类模型：

- **main model**：`OPENAI_MODEL`
- **light model**：`OPENAI_MODEL_LIGHT`；若未设置则回退到 main model

Paper memory 和 paper QA 直接使用 main model。Topic scan 会根据路由逻辑在 light model 和 main model 之间切换。这是当前主要的质量/成本控制机制。

<a id="how-it-works"></a>
## 工作原理

### 系统概览

整体数据流如下：

`paper -> chunks -> paper memory -> topic report -> topic memory -> daily digest`

更详细地说：

1. 从本地 PDF 或 arXiv id/URL ingest 一篇论文。
2. 提取文本并切成有重叠的 chunks。
3. 论文元数据和 chunks 写入 SQLite，抽取出的全文也会写到磁盘。
4. paper-memory prompt 将论文文本转成结构化 JSON memory。
5. Paper QA 使用该 memory 和检索出的 chunks 回答问题。
6. Topic scan 从外部 API 搜索候选论文，再用本地相关 paper memories 补强候选集。
7. Topic scan 写出 Markdown 报告，并在满足质量条件时把 JSON 快照持久化，供后续 topic memory 使用。
8. Daily digest 会为一个或多个 topic 搜索近期论文，并让 LLM 输出简洁的分诊式摘要。

### 存储布局

默认情况下，系统会写入 `./data/`：

- `data/research_copilot.db`: SQLite 数据库
- `data/papers/`: 本地 PDF 和提取出的文本
- `data/topics/`: 渲染后的 topic reports
- `data/digests/`: 渲染后的 digests
- `data/cache/`: ingest、paper memory、topic report 等 JSON cache artifacts

### 数据库实体

主要的 SQLite 表包括：

- `papers`
- `paper_chunks`
- `paper_memories`
- `topic_reports`
- `topic_report_versions`
- `subscriptions`
- `daily_digests`

Schema 刻意保持简单，方便本地数据可检查、可迁移、可长期维护。

<a id="setup"></a>
## 环境准备

```bash
cd research_copilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

至少需要设置：

```bash
OPENAI_API_KEY=...
```

可选配置包括：

- `OPENAI_MODEL`
- `OPENAI_MODEL_LIGHT`
- `SEMANTIC_SCHOLAR_API_KEY`
- `RESEARCH_COPILOT_DATA_DIR`
- `DATABASE_FILENAME`
- `HTTP_TIMEOUT`
- `LOG_LEVEL`

<a id="how-to-use"></a>
## 使用方式

主要交互入口是 CLI：

```bash
python cli.py --help
```

<a id="ingest-a-paper"></a>
### 1. Ingest 一篇论文

从 arXiv：

```bash
python cli.py ingest 1706.03762
python cli.py ingest https://arxiv.org/abs/1706.03762
python cli.py ingest 1706.03762 --with-memory
```

从本地 PDF：

```bash
python cli.py ingest /path/to/paper.pdf
```

这一步会把元数据写入 SQLite，把提取出的文本写到 `data/papers/`，并保存后续检索要用的 chunks。

默认情况下，`ingest` 保持为一个廉价、非 LLM 的操作。它会输出 `paper_id` 和建议的下一步命令，方便你立刻去构建 memory、后续提问，或再次从本地库里找到这篇论文。

如果你想走一步式流程，可以使用：

```bash
python cli.py ingest 1706.03762 --with-memory
```

这会先执行 ingest，再自动构建结构化 paper memory。

<a id="list-stored-papers"></a>
### 2. 列出已存储论文

```bash
python cli.py papers
python cli.py papers --limit 100
```

该命令会打印本地已存储论文，包含：

- `paper_id`
- 是否已经构建 memory
- external id（对于 arXiv 论文就是 arXiv id）
- year
- title

<a id="search-stored-papers"></a>
### 3. 搜索本地论文

```bash
python cli.py search-papers attention
python cli.py search-papers transformer
python cli.py search-papers 1706.03762
```

这会按标题文本、关键词、作者文本、venue/source 文本以及 external id 搜索本地论文。

<a id="build-paper-memory"></a>
### 4. 构建 paper memory

```bash
python cli.py memory 1
```

这里的 `1` 是 ingest 返回的 `papers.id`。

这一步会对 ingest 后的论文文本运行 paper-memory prompt，并将结果同时写入 SQLite 和 `data/cache/`。

<a id="ask-questions-about-a-paper"></a>
### 5. 围绕单篇论文提问

```bash
python cli.py ask 1 "What is the main contribution?"
python cli.py ask 1 "Why does the method work?"
python cli.py ask 1 --save "What is the main contribution?"
python cli.py ask-log 1
python cli.py ask-log 1 --tail 5
python cli.py ask-log-delete 1
python cli.py ask-log-delete 1 --index 3
python cli.py ask-log-delete 1 --question "slot attention"
```

这里会使用已存储的 paper memory 和从同一篇论文中检索出的文本 chunks。

如果你记不住数值型 `paper_id`，先运行 `python cli.py papers` 或 `python cli.py search-papers ...`。

默认情况下，`ask` 不会持久化回答。如果你想保留一个轻量的本地记录，可以加上 `--save`，把该轮问答追加写入：

`data/papers/qa/paper_<paper_id>.jsonl`

可以使用 `python cli.py ask-log <paper_id>` 直接打印这篇论文已保存的本地问答历史。

可以使用 `python cli.py ask-log <paper_id> --tail N` 只看最近 `N` 条已保存问答。

可以使用 `python cli.py ask-log-delete <paper_id>` 删除这篇论文对应的本地问答历史文件。

可以使用 `python cli.py ask-log-delete <paper_id> --index N` 按 `ask-log` 里显示的序号删除某一条问答。

可以使用 `python cli.py ask-log-delete <paper_id> --question "..."` 按问题字符串做大小写不敏感匹配，删除命中的问答。

`--save` 放在问题前后都可以被正确识别。

<a id="run-topic-scan"></a>
### 6. 运行 topic scan

```bash
python cli.py topic "diffusion models for video" --max-papers 25
```

它会：

- 搜索 arXiv 和 Semantic Scholar
- 先扩展并规范化 topic，再用多个子查询检索并合并结果
- 查找与该 topic 相关的本地 paper memories
- 生成结构化 topic report
- 将 Markdown 写入 `data/topics/`
- 将 JSON cache 写入 `data/cache/`
- 在质量启发式允许时，把版本化报告快照保存到 SQLite

### 使用高质量模式

```bash
python cli.py topic "test-time compute" --max-papers 30 --high-quality
```

`--high-quality` 会强制执行 topic-scan 的第二阶段 refinement，即使路由启发式原本会跳过它。

### 强制刷新 topic scan

```bash
python cli.py topic "4d object reconstruction / generation" --max-papers 20 --force
```

`--force` 会绕过这个 `(topic, max_papers)` 对应的 topic JSON cache，重新执行检索和 LLM synthesis。若不传 `--force`，当 cache version、topic 字符串和 `--max-papers` 都匹配时，CLI 会复用已有缓存。

<a id="topic-report-output"></a>
## Topic Report 输出

每次 topic scan 会写出：

- `data/topics/<slug>_report.md`：便于阅读的 Markdown 报告
- `data/cache/topic_report_<slug>.json`：包含结构化报告、检索摘要和候选附录的 JSON cache

### Markdown 主要部分

Topic report Markdown 一般包括：

- **Summary**：简洁的领域概览
- **Scan mode**：当前是 metadata-only 还是 memory-backed
- **Retrieval stats**：arXiv 数量、Semantic Scholar 数量、去重后数量、保留候选数、最终引用论文数、provider 失败或限流情况
- **Evidence quality**：明确说明证据强度和覆盖限制
- **Branches / subthemes**：topic 下的主要分支
- **Paper lists**：foundational、representative、recent valuable、lower-priority / possibly overhyped
- **Recent trends / Open questions**：基于候选集的综合结论
- **All retrieved candidate papers**：完整候选附录，包含标题、年份、来源、URL，以及可用时的 arXiv id

### Cache 行为

Topic scan 的缓存键由 topic slug 和 `--max-papers` 共同决定。JSON cache 中保存：

- 结构化 topic report
- retrieval query variants 和 provider stats
- 用于报告附录的保留候选论文列表

如果报告没有通过 SQLite 的持久化质量门槛，Markdown 和 JSON cache 仍然会写出；只是 topic-memory 快照不会入库。

### Provider 降级模式与限流

- 如果一个 provider 成功，另一个 provider 失败或被限流，topic scan 仍会以 degraded mode 继续，并把状态记录到 Markdown 的 retrieval 部分和 JSON cache 中。
- 如果两个 provider 都基本失败，且失败原因与 rate limit 有关，topic scan 会抛出 rate-limit 错误。
- Semantic Scholar 内部节流逻辑仍然保留；设置 `SEMANTIC_SCHOLAR_API_KEY` 可以提升配额并减少 degraded-mode 扫描。

<a id="generate-a-daily-digest"></a>
### 7. 生成 daily digest

`digest` 是“近期论文分诊”工作流。它会搜索某个时间窗口内发表的论文，按一个或多个 topic 组织，再让 LLM 输出简洁的推荐表，并把结果写入 `data/digests/`。

CLI 形式：

```bash
python cli.py digest [topics ...] [--days N] [--max-per-topic N]
python cli.py digest [topics ...] [--days N] [--max-per-topic N] [--debug-candidates]
```

参数：

- `topics`：0 个或多个 topic 字符串
- `--days`：回看最近多少天，默认 `3`
- `--max-per-topic`：每个 topic 在合并和 LLM 分诊前最多保留多少篇候选，默认 `15`
- `--debug-candidates`：额外写出一份 debug JSON，展示 relevance 和 recency 过滤前后的候选列表

行为：

- 如果显式传入一个或多个 `topics`，digest 只会对这些 topics 运行。
- 如果没有传入 `topics`，digest 会回退到 SQLite 中的 active subscriptions。
- 命令会把 digest Markdown 路径打印到 stderr，并把 `items: <count>` 打印到 stdout。
- digest 文件名现在会包含 UTC 日期、`days` 窗口和简短 topic 标签。
- 如果没有显式传 topic，文件名标签会写成 `subscribe`。

重要区别：

- `python cli.py subscriptions` 显示的是订阅的 digest topics，不是论文列表。
- `python cli.py digest ...` 才会实际执行近期论文检索，并在 Markdown 中生成论文列表。

`subscriptions` 输出列的含义：

- 第 1 列：topic slug，例如 `animal_dataset`
- 第 2 列：原始 topic 文本，例如 `animal dataset`
- 第 3 列：状态，通常是 `active` 或 `inactive`

因此下面这条命令：

```bash
python cli.py subscriptions
```

回答的是“如果我不显式传 topic，digest 会使用哪些 topics？”，而不是“已经找到了哪些论文？”

显式传 topic 运行 digest：

```bash
python cli.py digest --days 5 "reasoning models" "retrieval augmented generation"
```

等价示例：

```bash
python cli.py digest "animal motion"
python cli.py digest "animal motion" "4d generation" --days 7 --max-per-topic 20
```

也可以先订阅 topics，再在不显式传 topic 时复用它们：

```bash
python cli.py subscribe "reasoning models"
python cli.py subscribe "multimodal agents"
python cli.py subscriptions
python cli.py digest --days 3
```

示例工作流：

1. 订阅一个或多个长期关注的 topic：

```bash
python cli.py subscribe "animal dataset"
python cli.py subscribe "animal motion"
```

2. 查看当前哪些 topics 处于 active 状态：

```bash
python cli.py subscriptions
```

示例输出：

```text
animal_dataset  animal dataset  active
animal_motion   animal motion   active
```

这不代表论文已经被检索出来；只表示这些 topics 被保存成了 digest 的输入。

3. 在不传显式 topics 的情况下运行 digest：

```bash
python cli.py digest --days 3
```

这会使用上面的 active subscriptions，搜索最近 3 天的论文，并把 digest Markdown 写到 `data/digests/`。

4. 打开生成的 digest Markdown，查看实际的论文推荐。

如果你不想走 subscriptions，而是想直接搜索，可以显式传 topic：

```bash
python cli.py digest --days 3 "animal dataset"
python cli.py digest --days 7 "animal dataset" "animal motion"
```

Digest 内部会做的事情：

- 在合适的时候把 topic 扩展成轻量级 query variants，包括像 `rigging / articulation` 这样的斜杠输入
- 为每个 topic 或展开后的子查询检索近期论文
- 在最终按时间窗口保留前，先做一层轻量 lexical topic relevance 过滤
- 对跨 topic 重叠结果做合并和去重
- 把候选元数据送入 digest prompt
- 在 `data/digests/` 下写出 Markdown digest，例如 `digest_20260402_7d_animal_dataset_<hash>.md` 或 `digest_20260402_7d_subscribe_<hash>.md`

如果你传入 `--debug-candidates`，digest 还会额外写出一个同目录的 debug 文件，例如：

- `data/digests/digest_<stem>_debug.json`

这份 debug JSON 会按 topic 保存：

- 实际使用的 query variants
- relevance 过滤前的 merged candidates
- relevance 过滤后的 candidates
- recency 过滤后的 candidates
- 送入 LLM 之前最终保留的 candidates

Digest 不是什么：

- 不是 `python cli.py topic ...` 那种系统性 topic survey
- 不是对单篇论文的完整 review
- 默认情况下也不依赖本地 paper memory

更适合把它当成一个“先筛一遍最近值得读论文”的轻量过滤器。

<a id="design-philosophy"></a>
## 设计理念

### Local-first

系统把持久状态保存在本地：

- SQLite 存结构化记录
- 本地文件存提取文本、topic reports、digests 和各类 cache

这样整个工作流更容易检查、脚本化，也更适合 SSH 环境。

### 最少依赖

项目直接依赖的组件很少：

- Python
- SQLite
- `httpx`
- `pydantic`
- OpenAI SDK
- PDF extraction libraries

实现上有意避免引入沉重的 orchestration layer。

### Structured outputs

主要的 LLM 步骤都使用 Pydantic response schema：

- paper memory
- topic report
- daily digest

这样可以约束输出形状、简化验证，也便于把 JSON 干净地持久化下来。

### 尽量减少幻觉

系统主要通过以下方式降低幻觉：

- 用检索出的 chunks 约束 paper QA
- 约束 topic 和 digest 输出只能基于提供的候选论文
- 明确要求模型承认不确定性
- 用 schema 验证输出

这些手段可以降低风险，但不能彻底消除错误推理。

### 成本意识

系统遵循一个简单的成本策略：

- topic scan 第一阶段优先用 light model
- 只有必要时才升级到 main model
- 尽量保持检索为本地、词汇式逻辑
- 避免额外基础设施，例如向量服务

<a id="limitations"></a>
## 局限性

下面这些限制在真实研究场景中都很重要。

### Topic scan 部分依赖元数据

Topic scan 很大程度依赖标题、摘要、venue、日期，以及 arXiv 和 Semantic Scholar 的检索覆盖。只有一部分 topic 推理可以被本地完整 paper memories 支撑。

这意味着：

- 论文分类可能比较浅
- 趋势判断可能受检索偏差影响
- 缺失方向可能是 retrieval gap，不一定真的是领域空白

### LLM 输出不保证正确

Paper memory、QA、topic reports 和 digests 都依赖 LLM 判断。结构化输出验证只能检查格式，不能验证内容真实性。

### Topic memory 会积累偏差

历史 topic reports 会在后续 topic scan 中以简短 claim summary 的形式回灌。虽然 prompt 会要求模型把这些内容视为假设，但这仍然是一个自我引用的 memory loop，可能强化早期错误或偏置框架。

### Second-pass refinement 是启发式触发

第二阶段 topic refinement 由简单启发式触发，例如报告长度、grounded mentions 数量、prior memory 存在与否，以及显式的 `--high-quality`。这对成本控制有帮助，但不是严格的 epistemic quality 指标。

### Retrieval 是 lexical，不是 semantic

Paper QA 和 memory lookup 目前依赖 token overlap，而不是 embedding 或 learned ranking。只要措辞变化较大，系统就可能错过本来相关的证据。

### Digest 是 metadata triage，不是 paper review

Daily digest 的判断主要基于近期论文的元数据，尤其是标题和摘要。它应被视为“分诊建议”，而不是对技术质量或实验有效性的可靠评估。

默认情况下，digest 生成会优先用 `OPENAI_MODEL_LIGHT`；若未配置则回退到 `OPENAI_MODEL`。Digest 行里也可能包含一些轻量分诊字段，例如 `confidence`、`time_to_invest` 和 `signal_strength`。

### Ingest 质量依赖 PDF 文本抽取

如果 PDF extraction 很差，后续 memory 和 QA 的质量也会相应下降。

<a id="cost-considerations"></a>
## 成本考虑

主要开关包括：

- `OPENAI_MODEL`：用于 paper memory、paper QA 和 topic refinement 的主模型
- `OPENAI_MODEL_LIGHT`：topic scan 第一阶段和 digest 默认使用的轻量模型
- `--high-quality`：强制第二阶段 topic pass
- `--max-papers`：控制 topic scan 的检索广度
- `--days` 和 `--max-per-topic`：控制 digest 的规模

实用建议：

- 让 `OPENAI_MODEL_LIGHT` 比 `OPENAI_MODEL` 更便宜
- 先用默认 topic scan，再在必要时重跑 `--high-quality`
- 对 broad topic 保持适中的 `--max-papers`
- 把 digest 当作过滤器，而不是读论文的替代品

<a id="future-improvements"></a>
## 后续可改进方向

- 更强的 paper QA 和 topic-memory matching 检索
- 更严格地验证 topic claims 与支撑论文之间的对应关系
- 把 paper memory 中的证据跟踪更完整地传递到 topic reports
- 在 topic reports 中加入更明确的 confidence 字段
- 更好地处理差的 PDF extraction 和不完整元数据

<a id="tests"></a>
## 测试

```bash
python -m unittest discover -s tests -v
```

当前测试主要覆盖 parsing、chunking、database 行为、retrieval helpers、CLI wiring，以及 digest Markdown 渲染。它们并不能完整验证研究结论的正确性。
