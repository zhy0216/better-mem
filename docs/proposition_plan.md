# Proposition-Centric Memory Refactor Plan

> 从 `fact-centric` 记忆系统演进到 `proposition + belief + evidence` 的设计方案。

## 1. 背景与问题

当前系统以 `Fact` 作为最小持久化与检索单位，核心定义见现有设计文档与代码：

- `Fact` 被定义为单条、原子化、可独立检索的 proposition
- 写路径是 `messages -> facts -> profile`
- 读路径是 `query -> facts -> assembled context`

这个设计在工程上是成立的，因为它很好地服务了以下目标：

- 便于向量化与全文检索
- 便于打分、排序、去重和过期处理
- 便于冲突检测和软替代
- 便于按 query 即时组装 context

但从更高维度看，`Fact` 这个名字过于强，隐含了“这条内容已经是真实事实”的语义，而系统实际存储的往往只是：

- 用户或其他参与者说过的一条命题
- 模型从对话中抽取出的一条结构化表达
- 带有来源、时效性、可信度和适用范围的暂时性知识

因此，当前系统真正的最小单位并不是“事实”，而更接近：

- 一个命题 `proposition`
- 以及系统当前对该命题的信念程度 `belief`

进一步地，仅用“贝叶斯置信度”来命名最小单位也不准确，因为置信度描述的是系统对命题的态度，而不是被存储和检索的内容载体。

结论：

- `fact` 不是最准确的概念
- `confidence` 也不是最小单位
- 更合理的最小单位是 `proposition + belief`
- `evidence` 是支持 belief 的可追溯依据

## 2. 设计目标

本次重构的目标不是推翻现有 retrieval-first 架构，而是把底层语义从“事实库”升级为“信念驱动的命题库”。

目标如下：

1. 保留当前系统的高效检索、组装和画像生成能力
2. 显式表达不确定性，而不是默认所有存储项都是真实事实
3. 允许互斥命题在系统中并存，由证据和时间逐步竞争
4. 让来源、证据、置信度、重要性、时效性成为一等字段
5. 为未来的概率更新、溯源解释、冲突分析和 multi-source fusion 留出空间

非目标如下：

1. 不追求数学上严格完整的贝叶斯网络
2. 不在第一阶段引入复杂的因果图或全局概率图模型
3. 不把 reasoning / prediction 下沉到记忆层

## 3. 核心概念

### 3.1 Proposition

`Proposition` 是内容本体，描述“系统正在存储什么命题”。

例子：

- `张三计划 2024 年 4 月去东京`
- `张三喜欢 Vim`
- `张三目前居住在上海`

它回答的是：

- 这条记忆在说什么

### 3.2 Belief

`Belief` 是系统对 proposition 的当前信念状态。每个 proposition 在任何时刻只有**一个** active belief（1:1 关系）。

它回答的是：

- 系统现在有多相信这条 proposition
- 这条 proposition 对未来检索是否重要
- 这条 proposition 当前是否仍然活跃、过时或存在争议

注意：`belief` 不是“内容”，而是内容上的概率与状态层。

> **设计决策：为什么不把 belief 内联到 propositions？**
>
> 虽然 1:1 关系在理论上可以合并为一张表，但分表有以下好处：
>
> - belief 的更新频率远高于 proposition（每次新证据都会更新 confidence），分表避免频繁更新 propositions 行导致 embedding 索引失效
> - 未来如果需要保留 belief 变更历史（审计、回溯分析），可以自然扩展为 `belief_history` 表
> - 查询时可以通过 JOIN 或物化视图灵活组合
>
> 如果实测发现 JOIN 成为瓶颈，Phase 2 可以考虑合并或用物化视图。

### 3.3 Evidence

`Evidence` 是支撑或削弱某个 belief 的具体依据。

它回答的是：

- 为什么系统会相信或怀疑这条 proposition
- 证据来自谁、何时、哪条消息或哪个外部来源
- 这条证据是支持、反驳还是中性

### 3.4 Importance 与 Confidence 的区别

这是本设计里最关键的区分之一。

- `confidence`：命题为真的主观后验概率
- `importance`：即使命题不完全确定，它对未来 recall 是否仍然有价值

例子：

- “用户可能下周去东京” 的 `confidence` 可能只有 `0.58`
- 但如果用户正在制定行程，它的 `importance` 仍然可能很高

所以：

- `confidence` 不应替代 `importance`
- `importance` 也不应被误当成真实性概率

## 4. 新的最小记忆单位

新的最小记忆单位定义为：

```text
MemoryAtom = Proposition + Belief + Evidence*
```

其中：

- `Proposition` 是唯一内容载体
- `Belief` 是命题当前状态
- `Evidence` 是零到多条来源依据

这意味着：

- 系统存的不是“已被证明为真”的 fact
- 而是“可被检索、可被更新、可被证据支持或反驳的命题”

## 5. 数据模型

### 5.1 Proposition

建议模型：

```python
class Proposition(BaseModel):
    id: UUID
    tenant_id: str
    user_id: str
    group_id: str | None = None
    subject_id: str | None = None

    canonical_text: str
    proposition_type: Literal[
        "observation",   # third-party or inferred ("Zhang San was late today")
        "declaration",   # first-person assertion ("I am a backend engineer")
        "plan",          # future intent ("Zhang San plans to visit Tokyo")
        "preference",    # likes / dislikes ("Zhang San prefers Vim")
        "relation",      # interpersonal ("Zhang San is Li Si's colleague")
    ]

    semantic_key: str | None = None
    tags: list[str] = []
    metadata: dict = {}

    valid_from: datetime | None = None
    valid_until: datetime | None = None
    first_observed_at: datetime | None = None
    last_observed_at: datetime | None = None

    embedding: list[float] | None = None
    created_at: datetime
    updated_at: datetime
```

字段说明：

- `canonical_text`：命题的规范化文本，用于展示、索引和组装
- `proposition_type`：命题类型，保留 `observation` / `declaration` 的区分——二者来源可靠性差异很大（用户亲口声明 vs 第三方观察），在 belief 初始 prior 赋值时需要区别对待
- `subject_id`：命题的主体。在多人对话中，命题的"关于谁"可能与 `user_id`（对话参与者）不同。例如"李四说张三下周要出差"，`user_id` 是李四，`subject_id` 是张三。默认为 `None` 表示与 `user_id` 相同
- `semantic_key`：命题所属语义槽位，用于竞争更新和冲突消解（详见第 6 节）
- `first_observed_at` / `last_observed_at`：首尾证据时间

### 5.2 Belief

建议模型：

```python
class Belief(BaseModel):
    id: UUID
    proposition_id: UUID  # UNIQUE — 每个 proposition 只有一个 active belief

    confidence: float
    prior: float
    source_reliability: float

    utility_importance: float
    freshness_decay: float

    support_count: int = 0
    contradiction_count: int = 0

    access_count: int = 0
    last_accessed: datetime | None = None

    status: Literal["active", "uncertain", "stale", "deprecated"]

    created_at: datetime
    updated_at: datetime
```

字段说明：

- `proposition_id`：与 proposition 的 1:1 关系，有 UNIQUE 约束
- `confidence`：当前后验置信度，范围 `0~1`
- `prior`：命题初始先验，由 `proposition_type` 和来源决定（详见第 7.2 节）
- `source_reliability`：该 belief 的来源整体可信度估计
- `utility_importance`：检索价值，承接当前 `importance`
- `freshness_decay`：时效衰减参数，承接当前 `decay_rate`，按 `proposition_type` 差异化设置（preference/relation 低衰减，plan 高衰减）
- `status`：当前可用状态，不再只表达替代关系

### 5.3 Evidence

建议模型：

```python
class Evidence(BaseModel):
    id: UUID
    proposition_id: UUID  # belief 通过 proposition 1:1 关系自动关联，无需冗余 FK

    evidence_type: Literal["utterance", "observation", "import", "inference"]
    direction: Literal["support", "contradict", "neutral"]

    source_type: str
    source_id: str | None = None
    source_meta: dict | None = None

    speaker_id: str | None = None
    quoted_text: str | None = None
    observed_at: datetime | None = None

    weight: float
    metadata: dict = {}

    created_at: datetime
```

字段说明：

- `proposition_id`：只挂 proposition，不再冗余引用 belief。由于 belief 与 proposition 是 1:1，通过 proposition_id 即可关联到对应 belief，避免两个 FK 指向不一致的数据完整性风险
- `direction`：证据方向，支持或反驳
- `weight`：该证据对 belief 的影响强度（量纲参考见第 7.2 节）
- `quoted_text`：原始文本片段，便于审计与解释

## 6. `semantic_key` 设计

### 6.1 为什么需要 `semantic_key`

如果只有 `canonical_text` 和 embedding，系统只能知道两条命题"像不像"，但不知道它们是否在竞争同一个语义槽位。

`semantic_key` 用来表达"这些 proposition 在回答同一个问题"。

例如：

- `residence`
- `favorite_editor`
- `trip_tokyo_2024_04.budget`
- `current_employer`

> 注意：`semantic_key` 不包含 `user_id` 前缀，用户隔离由查询条件保证。

当新命题进入系统时：

- 如果没有 `semantic_key`，只能做相似度搜索
- 如果有 `semantic_key`，就可以在同槽位内做竞争、衰减和置信更新

这一步是从"文本检索系统"升级到"语义记忆系统"的关键。

### 6.2 生成策略

`semantic_key` 由 LLM 在 extraction 阶段生成，作为 proposition 输出的一部分。

生成 prompt 的关键指令：

```text
For each proposition, generate a `semantic_key` that represents the semantic slot
this proposition is answering. Rules:

1. Use dot-separated lowercase English, e.g. "residence", "favorite_editor",
   "trip_tokyo_2024_04.budget"
2. The key should describe WHAT QUESTION this proposition answers,
   not the answer itself. "residence" is correct, "lives_in_shanghai" is wrong.
3. For time-scoped topics, include a time qualifier: "trip_tokyo_2024_04",
   not just "trip"
4. For stable attributes (preferences, skills, relations), use simple keys:
   "preferred_language", "employer"
5. If you cannot confidently determine the semantic slot, output null
```

### 6.3 一致性保障

LLM 生成 `semantic_key` 的最大风险是不稳定——不同时间、不同措辞可能产出不同 key。缓解措施：

1. **候选归一**：新 proposition 生成 key 后，先查询同 user 下已有的 `semantic_key` 列表，用 LLM 二次判断是否应该复用已有 key 而非创建新 key
2. **Key 规范化**：对生成的 key 做后处理——lowercase、去多余空格、统一分隔符为 `.`
3. **Embedding fallback**：如果 `semantic_key` 为 null 或未命中已有 key，退回到 embedding 相似度做候选查找（即现有路径，不会比当前更差）

### 6.4 Fallback 行为

`semantic_key` 是可选字段，生成失败不应阻塞写入：

- key 为 null 的 proposition 仍然正常存储和检索
- 冲突检测退化为纯 embedding 相似度模式（与当前系统行为一致）
- 后台可定期对 key 为 null 的 proposition 做补填（batch job）

### 6.5 已有数据回填

Phase 1 迁移时，旧 `facts` 转 `propositions` 的 `semantic_key` 可以：

1. 先全部设为 null
2. 后台跑一次 batch job，用 LLM 为每条 proposition 生成 key
3. 对同 user 下相似 key 做去重归一
4. 回填完成后再开启基于 `semantic_key` 的槽位竞争逻辑

## 7. 概率语义

### 7.1 理想定义

更严谨的表达应该是：

```text
belief.confidence = P(proposition | evidence, context, time)
```

也就是：

- 给定证据
- 给定上下文
- 给定当前时间
- 系统对 proposition 为真的后验概率

### 7.2 工程实现建议

第一阶段不建议引入严格完整的贝叶斯图模型，而采用可解释、可落地的近似更新方式：

```text
logit(posterior)
= logit(prior)
+ sum(support_evidence_weight)
- sum(contradict_evidence_weight)
- time_decay_factor
```

最终：

```text
confidence = sigmoid(logit(posterior))
```

这比简单覆盖或硬替代更合理，因为：

- 新证据是更新 belief，而不是直接抹掉旧命题
- 互斥 proposition 可以并存
- “用户亲口说过” 与 “第三方转述” 可以赋不同权重
- 长时间未被支持的 proposition 可以自然衰减

#### 7.2.1 Prior 参考值

初始 prior 由 `proposition_type` 和来源决定：

| proposition_type | 来源为 declaration | 来源为 observation / inference |
| --- | --- | --- |
| declaration | 0.85 | 0.60 |
| preference | 0.80 | 0.55 |
| relation | 0.75 | 0.50 |
| observation | 0.65 | 0.50 |
| plan | 0.60 | 0.45 |

用户亲口声明的 prior 显著高于第三方转述或模型推断。

#### 7.2.2 Evidence Weight 量纲参考

| evidence_type | 建议 weight | 说明 |
| --- | --- | --- |
| utterance (本人) | 1.0 | 用户亲口说的，最强信号 |
| utterance (第三方) | 0.6 | 他人转述，可信度打折 |
| observation | 0.7 | 系统或外部观察到的行为 |
| import | 0.8 | 从可信外部源导入 |
| inference | 0.4 | 模型推断，最低权重 |

weight 在 logit space 的含义：weight=1.0 会把 0.5 的 prior 推到约 0.73；weight=0.4 推到约 0.60。这个幅度意味着通常需要 2-3 条独立证据才能把 confidence 推到 0.9 以上，符合"渐进确认"的设计意图。

#### 7.2.3 Time Decay 差异化

`time_decay_factor` 不应对所有 proposition_type 一视同仁（承接现有代码 `_DECAY_RATES` 的设计）：

```text
time_decay_factor = freshness_decay * age_days
```

| proposition_type | freshness_decay | 含义 |
| --- | --- | --- |
| plan | 0.05 | 计划时效性强，快速衰减 |
| observation | 0.02 | 观察中等衰减 |
| declaration | 0.005 | 用户声明相对稳定 |
| preference | 0.005 | 偏好长期有效 |
| relation | 0.003 | 关系最稳定 |

注意：`time_decay_factor` 的上界应被 clamp，防止长期无新证据的稳定偏好（如"喜欢 Vim"）confidence 降到接近 0。建议：

```text
time_decay_factor = min(freshness_decay * age_days, max_decay_cap)
```

其中 `max_decay_cap` 建议为 `2.0`（对应 confidence 下限约 0.12），确保即使完全无新证据，稳定命题也不会被彻底遗忘。

#### 7.2.4 更新时机

Belief 更新在 **evidence 写入时同步触发**：

1. 新 evidence 插入 `evidence` 表
2. 立即重算对应 proposition 的 `confidence`：遍历该 proposition 下所有 evidence 的 weight 和 direction，加上 time_decay
3. 更新 `beliefs` 表的 `confidence`、`support_count`、`contradiction_count`、`updated_at`

这保证了 belief 状态始终是最新的，不会出现"证据已写入但 confidence 尚未更新"的不一致窗口。

### 7.3 如果不做真贝叶斯

如果第一阶段还不想引入 `prior` / `logit` 这套语义，也可以先诚实地把字段命名为：

- `belief_score`
- `source_reliability`
- `uncertainty`

避免把一个纯启发式打分系统误称为贝叶斯系统。

## 8. 检索与排序

当前系统的排序逻辑可以概括为：

```text
vector_search + keyword_search
-> RRF merge
-> importance * recency * access boost
```

在 proposition 模型下，建议改成加权求和（而非纯乘法），避免任一因子接近 0 时压死整个分数：

```text
retrieval_score
= w1 * semantic_relevance
+ w2 * belief_confidence
+ w3 * utility_importance
+ w4 * freshness_factor
+ w5 * access_boost
```

建议初始权重与各项含义：

| 因子 | 权重 | 范围 | 回答的问题 |
| --- | --- | --- | --- |
| `semantic_relevance` | w1 = 0.40 | 0~1 (RRF 归一化后) | 像不像——查询与 proposition 内容的语义相关度 |
| `belief_confidence` | w2 = 0.25 | 0~1 | 真不真——系统是否相信它为真 |
| `utility_importance` | w3 = 0.20 | 0~1 | 值不值——它对未来 recall 的价值 |
| `freshness_factor` | w4 = 0.10 | 0~1 (exp decay) | 新不新——在当前时间点是否仍适用 |
| `access_boost` | w5 = 0.05 | 1.0 + 0.1*log1p(n), 归一化到 0~1 | 常不常用——历史访问正反馈 |

> **为什么不用纯乘法？** 纯乘法 `a * b * c * d * e` 中，任何一项为 0 或接近 0 就会把整个分数压死。例如一个全新的高相关、高置信 proposition（`access_boost` 低）会被不合理地排到后面。加权求和保证每个维度独立贡献，不会互相"拖累"。

> **调参建议：** 以上权重为初始建议值，应通过离线评测（如 NDCG@10 on recall test set）进行调优。

## 9. 写入与更新流程

建议写入路径升级为：

```text
messages
-> proposition extraction
-> proposition normalization
-> semantic_key generation
-> evidence creation
-> candidate proposition lookup
-> belief update / competing proposition insertion
-> profile synthesis
```

### 9.1 Extraction

LLM 不再输出 `facts`，而输出 `propositions`：

- 命题文本
- proposition 类型
- semantic_key 候选
- valid 时间范围
- 初始 prior / confidence 建议值
- tags / metadata

### 9.2 Normalization

对命题做标准化：

- 代词消解
- 时间绝对化
- 同义归一
- 命名实体规范化
- 文本 canonicalization

### 9.3 Evidence Creation

每条输入消息都转成一条或多条 evidence，挂在 proposition 下。

### 9.4 Candidate Lookup

按以下顺序查找候选 proposition：

1. 精确 `semantic_key`
2. 高相似 embedding
3. 规则补充召回

### 9.5 Belief Update

如果命中同一 proposition：

- 追加 evidence
- 更新 `support_count` / `contradiction_count`
- 更新 `confidence`
- 更新 `last_observed_at`

如果命中同一 `semantic_key` 下的不同 proposition：

- 视为竞争命题
- 各自保留
- 由新证据更新各自 belief

如果没有命中：

- 创建新 proposition
- 创建对应 belief
- 写入首条 evidence

## 10. 冲突处理

当前系统的冲突处理是：

- 找到相似 facts
- 判断 `contradicts` / `updates`
- 用 `superseded_by` 建立软替代链

这在 `fact-centric` 模型下是合理的，但在 `belief-centric` 模型下应该改为：

### 10.1 冲突不等于删除

冲突意味着：

- 某条 proposition 的 belief 被削弱
- 或者另一个 proposition 的 belief 被增强

而不是立即删除旧 proposition。

### 10.2 槽位竞争

对同一 `semantic_key` 下互斥的 proposition：

- 允许短期共存
- 分别维护各自 `confidence`
- 当某条 proposition 明显占优时，将其他 proposition 标记为 `deprecated` 或 `uncertain`

### 10.3 可解释性

最终回答“为什么系统认为用户住在上海而不是北京”时，可以回溯：

- 支持上海的证据
- 反驳北京的证据
- 两者各自的 posterior 变化

这比 `superseded_by` 更符合真实记忆系统的演化过程。

## 11. 对 Profile 的影响

当前 profile 是 facts 的上层派生物，这个原则应该保留。

但 profile 生成逻辑应从“读取最新 facts”升级为“读取高 confidence、且高 utility 的 propositions”。

建议规则：

- profile summary 只使用 `confidence >= threshold` 的 propositions
- 对易变槽位，如位置、公司、计划，优先使用最近且高置信 proposition
- 对长期特征，如偏好、技能、关系，可汇总多个 evidence 支撑的 proposition

这样 profile 会更稳定，也更容易解释。

## 12. 建议的数据库表设计

### 12.1 `propositions`

```sql
CREATE TABLE propositions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id           TEXT NOT NULL DEFAULT 'default',
    user_id             TEXT NOT NULL,
    group_id            TEXT,
    subject_id          TEXT,          -- 命题主体，默认 NULL 表示与 user_id 相同

    canonical_text      TEXT NOT NULL,
    proposition_type    TEXT NOT NULL,
        -- observation: 第三方或推断 ("张三今天迟到了")
        -- declaration: 用户第一人称声明 ("我是后端工程师")
        -- plan:        未来计划 ("张三计划下周去东京")
        -- preference:  偏好 ("张三喜欢 Vim")
        -- relation:    关系 ("张三是李四的同事")
    semantic_key        TEXT,

    valid_from          TIMESTAMPTZ,
    valid_until         TIMESTAMPTZ,
    first_observed_at   TIMESTAMPTZ,
    last_observed_at    TIMESTAMPTZ,

    embedding           vector(1024),
    tsv                 tsvector GENERATED ALWAYS AS (
                            to_tsvector('simple', canonical_text)
                        ) STORED,

    tags                TEXT[] DEFAULT '{}',
    metadata            JSONB DEFAULT '{}',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

建议索引：

- `(tenant_id, user_id, semantic_key)`
- `embedding` HNSW
- `tsv` GIN
- `(tenant_id, user_id, last_observed_at DESC)`

### 12.2 `beliefs`

```sql
CREATE TABLE beliefs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposition_id      UUID NOT NULL UNIQUE
                            REFERENCES propositions(id) ON DELETE CASCADE,

    confidence          FLOAT NOT NULL DEFAULT 0.5,
    prior               FLOAT NOT NULL DEFAULT 0.5,
    source_reliability  FLOAT NOT NULL DEFAULT 0.8,

    utility_importance  FLOAT NOT NULL DEFAULT 0.5,
    freshness_decay     FLOAT NOT NULL DEFAULT 0.01,

    support_count       INT NOT NULL DEFAULT 0,
    contradiction_count INT NOT NULL DEFAULT 0,

    access_count        INT NOT NULL DEFAULT 0,
    last_accessed       TIMESTAMPTZ,

    status              TEXT NOT NULL DEFAULT 'active',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

建议索引：

- `proposition_id` 已有 UNIQUE 隐式索引
- `(status, confidence DESC)`
- `(utility_importance DESC)`

### 12.3 `evidence`

```sql
CREATE TABLE evidence (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposition_id      UUID NOT NULL REFERENCES propositions(id) ON DELETE CASCADE,
    -- belief 通过 proposition 1:1 关系自动关联，无需冗余 FK

    evidence_type       TEXT NOT NULL,
        -- utterance:   用户或参与者的原始发言
        -- observation: 系统或外部观察到的行为
        -- import:      从外部可信源导入
        -- inference:   模型推断
    direction           TEXT NOT NULL DEFAULT 'support',
        -- support / contradict / neutral

    source_type         TEXT NOT NULL,
    source_id           TEXT,
    source_meta         JSONB,

    speaker_id          TEXT,
    quoted_text         TEXT,
    observed_at         TIMESTAMPTZ,

    weight              FLOAT NOT NULL DEFAULT 1.0,
    metadata            JSONB DEFAULT '{}',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

建议索引：

- `(proposition_id, observed_at DESC)`
- `(source_type, source_id)`

## 13. 与当前 `facts` 模型的字段映射

| 当前 `facts` 字段 | 新归属 | 说明 |
| --- | --- | --- |
| `content` | `propositions.canonical_text` | 命题内容 |
| `fact_type` | `propositions.proposition_type` | 类型迁移 |
| `embedding` | `propositions.embedding` | 向量不变 |
| `valid_from` / `valid_until` | `propositions.valid_*` | 时效不变 |
| `occurred_at` | `evidence.observed_at` | 变成证据时间 |
| `source_type` / `source_id` / `source_meta` | `evidence.*` | 来源下沉到 evidence |
| `speaker_id` | `evidence.speaker_id` | 谁说的变成证据属性 |
| `importance` | `beliefs.utility_importance` | 保留检索价值信号 |
| `decay_rate` | `beliefs.freshness_decay` | 保留时效衰减 |
| `access_count` / `last_accessed` | `beliefs.*` | 保留访问反馈 |
| `superseded_by` / `supersedes` | 竞争 proposition + belief 状态 | 不再作为核心结构 |
| `status` | `beliefs.status` | 从事实状态变成信念状态 |
| `tags` / `metadata` | 优先放 `propositions.*` | 必要时 evidence 也可带 metadata |

## 14. API 影响

### 14.1 写接口

- `POST /v1/memorize` 返回 `propositions`（直接替换，不保留 `facts` 字段名）

### 14.2 查接口

- `GET /v1/propositions` 替代 `GET /v1/facts`
- `POST /v1/recall` 检索 proposition，响应附带 belief 与 top evidence

### 14.3 管理接口

- `POST /v1/propositions/{id}/evidence` — 证据注入
- `PATCH /v1/beliefs/{id}` — 人工修正
- `GET /v1/propositions/{id}/evidence` — 审计与解释
- `DELETE /v1/propositions/{id}` — 软删除

## 15. 迁移方案

> 本项目不需要向后兼容，因此采用直接替换策略，不保留 `facts` 表和旧接口。

### 15.1 Phase 1: 建表与数据迁移

1. 创建 `propositions`、`beliefs`、`evidence` 三张新表及索引
2. 运行迁移脚本，将旧 `facts` 数据转入新表：
   - 每条旧 `fact` → 一条 `proposition`
   - 同时创建一条默认 `belief`（prior / confidence 基于 `fact_type` 查表赋值）
   - 以原始来源信息创建一条 `evidence`
3. 后台 batch job 为每条 proposition 生成 `semantic_key`（详见第 6.5 节）
4. 验证完成后 `DROP TABLE facts CASCADE`

**验收标准**：

- 每条 proposition 恰好有一条 belief 和至少一条 evidence
- 新表索引创建完成，`EXPLAIN ANALYZE` 确认查询走索引
- `semantic_key` 回填覆盖率 >= 80%

### 15.2 Phase 2: 代码全量替换

直接替换所有读写路径：

1. `extract_facts` → `extract_propositions`（含 semantic_key 生成）
2. `detect_contradictions` → `update_beliefs`（基于 evidence 的 belief 更新）
3. `fact_store` → `proposition_store`（读写均指向新表）
4. `ranker` → 使用新排序公式（第 8 节）
5. `profile synthesizer` → 基于高 confidence propositions
6. API 端点全部切换为 `/v1/propositions`、`/v1/recall` 等

**验收标准**：

- 全部测试通过（包括更新后的测试用例）
- P99 检索延迟 < 200ms
- profile 质量人工抽检无明显下降

### 15.3 Phase 3: 清理

- 删除旧的 `Fact`、`ScoredFact`、`ContradictionPair` 等模型类
- 删除 `src/store/fact_store.py`、`src/extract/contradiction.py` 等旧文件
- 删除 `facts` 相关的迁移脚本（保留新表的迁移脚本）
- 更新 README 和 API 文档

**验收标准**：代码中不再有 `fact` 相关引用（测试除外）；CI 全绿。

## 16. 实施优先级

建议优先级如下：

1. 先定义 `semantic_key`
2. 再拆分 `proposition / belief / evidence`
3. 然后重写冲突处理为 belief update
4. 最后再调整 recall 排序与 profile synthesis

原因：

- 没有 `semantic_key`，belief 更新只能基于文本相似度，精度不稳定
- 先拆表不先定义槽位，会导致结构升级但语义能力没有本质提升

## 17. 风险与权衡

### 17.1 风险

- LLM 抽取 `semantic_key` 可能不稳定
- proposition 归一化不充分会造成重复命题
- belief 更新如果参数不当，容易出现置信漂移
- 检索排序维度变多后，调参成本会上升

### 17.2 权衡

- 现有 `fact-centric` 模型更简单，工程风险更低
- 新模型更准确，但数据结构、读写流程和调参复杂度都会增加

由于本项目不需要向后兼容，采用直接替换策略：

- 一次性建表、迁移数据、全量替换代码
- 迁移完成后清理所有旧 fact 相关代码

## 18. 代码改动清单

以下是按模块列出的预估改动范围，供任务拆分参考：

### 18.1 数据模型层 (`src/models/`)

| 文件 | 改动 | 预估 |
| --- | --- | --- |
| `src/models/fact.py` | 替换为 `proposition.py`；定义 `Proposition`、`Belief`、`Evidence`、`BeliefUpdateCandidate` 模型类；删除 `Fact`、`ScoredFact`、`ContradictionPair` | 中 |
| `src/models/api.py` | recall 响应增加 `belief` 和 `top_evidence` 可选字段 | 小 |

### 18.2 抽取层 (`src/extract/`)

| 文件 | 改动 | 预估 |
| --- | --- | --- |
| `src/extract/fact_extractor.py` | 重命名为 `proposition_extractor.py`；LLM prompt 输出增加 `semantic_key`、`prior`、`proposition_type`；输出类型改为 `PropositionCreate` | 大 |
| `src/extract/prompts.py` | 更新 extraction prompt（增加 semantic_key 生成指令）；更新 contradiction prompt 为 belief update prompt | 中 |
| `src/extract/contradiction.py` | 重构为 `belief_updater.py`；从"找矛盾 → 标记 superseded"变为"找候选 → 追加 evidence → 更新 confidence"；增加 semantic_key 精确匹配路径 | 大 |

### 18.3 存储层 (`src/store/`)

| 文件 | 改动 | 预估 |
| --- | --- | --- |
| `src/store/fact_store.py` | 替换为 `proposition_store.py`；实现 `insert_proposition`（含 belief + evidence 一起写入）、`update_belief`、`add_evidence`；搜索函数读 propositions JOIN beliefs；删除旧 `fact_store.py` | 大 |
| `src/migrations/` | 新增迁移脚本：创建三张新表 + 索引；数据迁移脚本（facts → propositions + beliefs + evidence） | 中 |

### 18.4 检索层 (`src/retrieve/`)

| 文件 | 改动 | 预估 |
| --- | --- | --- |
| `src/retrieve/ranker.py` | 排序公式从 `importance * recency * access_boost` 改为加权求和（第 8 节） | 小 |
| `src/retrieve/assembler.py` | context assembly prompt 增加 confidence 信息；可选附带 top evidence | 小 |

### 18.5 API 层 (`src/api/`)

| 文件 | 改动 | 预估 |
| --- | --- | --- |
| `src/api/memorize.py` | 内部调用从 `extract_facts` 改为 `extract_propositions` | 中 |
| `src/api/facts.py` | 替换为 `propositions.py`；端点改为 `/v1/propositions`；删除旧 `/v1/facts` | 中 |
| `src/api/recall.py` | 响应可选附带 belief + evidence | 小 |

### 18.6 测试 (`tests/`)

| 文件 | 改动 | 预估 |
| --- | --- | --- |
| `tests/test_extract/` | 更新 extraction 测试（输出结构变化）；新增 semantic_key 生成测试；重写 contradiction 测试为 belief update 测试 | 中 |
| `tests/test_store/` | 新增 proposition_store 测试（CRUD + belief update + evidence 追加） | 中 |
| `tests/test_retrieve/` | 更新 ranker 测试（新排序公式）；更新 assembler 测试 | 小 |
| `tests/test_api/` | 更新 memorize / recall 集成测试 | 小 |

> **总体预估**：约 15-20 个文件需要修改或新增，核心大改动集中在 extraction、contradiction → belief_update、store 三个模块。

## 19. 最终结论

从高维度看，当前系统把 `fact` 当作最小记忆单位，在工程上是正确的，但在概念上是不够精确的。

更合理的表述应该是：

- 最小内容单位：`proposition`
- 最小状态单位：`belief`
- 最小可追溯依据：`evidence`

因此，未来系统应从：

```text
fact-centric memory
```

演进为：

```text
proposition-centric, belief-aware memory
```

这是一次“语义升级”，不是对 retrieval-first 路线的否定。

底层仍然服务于检索，但不再假装每条存储项都是绝对事实，而是明确承认：

- 记忆本质上是被证据支持的命题
- 命题的地位会随时间和新证据变化
- 记忆系统管理的不是 facts，而是 beliefs over propositions
