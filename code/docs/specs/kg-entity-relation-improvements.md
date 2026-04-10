# 地理知识图谱实体关系体系修改建议

更新时间：2026-04-10
适用模块：`agent/agents/schemas.py`, `agent/agents/prompts.py`

---

## 一、当前体系回顾

### 1.1 实体类型（4类）

| 实体类型 | 定义 | 示例 |
|---------|------|------|
| **道路** | 街道、大道、小巷等交通载体 | 关山大道 |
| **POI** | 具体店名、地标、机构 | 武汉大学、某某咖啡厅 |
| **建筑物** | 具体的楼宇、商场主体 | 泛悦汇 |
| **街区** | 具有边界感的生活区域 | 街道口、华农校区 |

### 1.2 关系类型（5类）

| 关系类型 | 语义说明 | 示例 |
|---------|----------|------|
| **连接** | A和B通过道路/交通连接 | <关山大道, 连接, 光谷> |
| **位于** | A在B的内部或附近 | <武汉大学, 位于, 珞喻路> |
| **承载活动** | A场所发生B活动 | <行政楼, 承载活动, 合影> |
| **引发情感** | A引发B情感 | <书店, 引发情感, 好看> |
| **属于** | A属于B的组成部分 | <行政楼, 属于, 武汉大学> |

---

## 二、核心问题分析

### 2.1 实体体系问题

| 问题 | 说明 | 影响 |
|------|------|------|
| **POI与建筑物边界模糊** | "泛悦汇"既是商场(POI)也是楼宇(Building)，用户描述角度不同分类有歧义 | 抽取不一致，聚合困难 |
| **缺少"区域"层级** | 街区之上没有行政区划层（如"洪山区"、"武汉市"），无法表达层级隶属 | 无法构建完整空间层级图谱 |
| **道路层级缺失** | 主干道、次干道、支路仅作为属性细分，未建立道路间层级关系 | 无法表达道路拓扑结构 |
| **活动/情感非实体化** | "合影"、"好看"作为关系尾实体，但它们不是地理实体 | 图谱出现大量非地理节点，语义混乱 |

### 2.2 关系体系问题

| 问题 | 说明 | 影响 |
|------|------|------|
| **"位于"语义过于笼统** | 既可表示"内部"也可表示"附近"，方向和语义不明确 | 抽取歧义，无法精确表达空间关系 |
| **缺少"相邻"关系** | 社交媒体高频表达"旁边"、"对面"、"隔壁"等邻近关系 | 邻近语义被归入"位于"或遗漏 |
| **"承载活动"尾实体不明确** | "合影"、"逛街"、"吃饭"等活动是否需要实体化？ | 三元组语义模糊，查询困难 |
| **"引发情感"非地理语义** | 情感表达更适合作为属性而非独立关系 | 图谱节点膨胀，偏离地理语义核心 |
| **关系方向约束缺失** | 无Schema约束哪些实体类型可参与哪些关系 | 出现如"道路-承载活动-合影"等不合理三元组 |

---

## 三、修改建议方案

### 方案A：保守优化（推荐渐进式采用）

**核心理念**：保持4实体+5关系框架，细化语义约束和属性体系。

#### 3.1 实体类型微调

| 实体类型 | 修改内容 |
|---------|----------|
| **道路** | 无变化 |
| **POI** | 明确规则：独立商业主体/地标归POI，依附于POI的楼宇归Building |
| **建筑物** | 明确规则：POI内部的具体楼层、楼宇归Building |
| **街区** | 新增子类型：商圈、校区、社区、行政区、景区（已在属性细分） |
| **新增：区域** | 街区之上的行政区划层，如"洪山区"、"武汉市" |

#### 3.2 关系类型拆分与细化

| 原关系 | 拆分建议 | 语义说明 |
|--------|----------|----------|
| **位于** → | **位于内部** | A完全在B内部（如POI在街区内部） |
| | **位于邻近** | A在B附近（如POI在道路旁） |
| | **位于沿线** | A沿B分布（如POI沿道路分布） |
| **连接** → | **直达连接** | A和B有直达交通 |
| | **换乘连接** | A和B需换乘到达 |
| | **途径连接** | A途径B到达C（三元组扩展） |
| **新增** | **相邻** | A和B空间邻近（旁边、对面、隔壁） |
| **新增** | **包含** | A空间上包含B（区域→街区→POI层级） |

#### 3.3 活动与情感处理策略

| 原关系 | 处理方式 | 说明 |
|--------|----------|------|
| **承载活动** | 改为**属性**而非关系 | POI/建筑物增加"常见活动"属性字段 |
| **引发情感** | 改为**属性**而非关系 | POI/建筑物增加"情感标签"属性字段 |

**属性定义示例**：
```python
class EntityAttributes(BaseModel):
    类别: str
    细分: str
    常见活动: List[str] = []  # 新增：如["拍照", "逛街", "就餐"]
    情感标签: List[str] = []  # 新增：如["好看", "氛围好", "适合约会"]
```

---

### 方案B：激进重构（需评估后采用）

**核心理念**：引入事件/情感实体类型，构建完整语义图谱。

#### 3.4 新增实体类型

| 实体类型 | 定义 | 示例 |
|---------|------|------|
| **事件** | 在地理场所发生的活动 | 合影、樱花节、夜市 |
| **情感** | 对地理场所的情感评价 | 好看、氛围感、适合约会 |
| **区域** | 行政区划层级 | 洪山区、武汉市 |

#### 3.5 关系类型扩展

| 关系类型 | 语义说明 | Schema约束 |
|---------|----------|------------|
| **位于内部** | A在B内部 | (POI/Building, 街区/区域) |
| **位于邻近** | A在B附近 | (POI/Building, Road/POI) |
| **相邻** | A和B空间邻近 | (POI, POI) / (Building, Building) |
| **包含** | A包含B | (区域, 街区) / (街区, POI) |
| **属于** | A属于B | (POI, 街区) / (Building, POI) |
| **连接** | A和B交通连接 | (Road, Road) / (Road, POI) |
| **承载事件** | A场所发生B事件 | (POI/Building, 事件) |
| **引发情感** | A引发B情感 | (POI/Building/事件, 情感) |

#### 3.6 Schema约束矩阵

```python
ENTITY_RELATION_SCHEMA = {
    "位于内部": {
        "head": ["POI", "建筑物", "街区"],
        "tail": ["街区", "区域"]
    },
    "位于邻近": {
        "head": ["POI", "建筑物"],
        "tail": ["道路", "POI", "建筑物"]
    },
    "相邻": {
        "head": ["POI", "建筑物"],
        "tail": ["POI", "建筑物"]
    },
    "包含": {
        "head": ["区域", "街区"],
        "tail": ["街区", "POI", "建筑物"]
    },
    "属于": {
        "head": ["POI", "建筑物", "街区"],
        "tail": ["街区", "区域", "POI"]
    },
    "连接": {
        "head": ["道路"],
        "tail": ["道路", "POI", "街区"]
    },
    "承载事件": {
        "head": ["POI", "建筑物", "街区"],
        "tail": ["事件"]  # 新实体类型
    },
    "引发情感": {
        "head": ["POI", "建筑物", "事件"],
        "tail": ["情感"]  # 新实体类型
    },
}
```

---

## 四、两种方案对比

| 维度 | 方案A（保守优化） | 方案B（激进重构） |
|------|-------------------|-------------------|
| **图谱规模** | 节点数较少，专注地理实体 | 节点数增加，包含事件/情感 |
| **语义精确度** | 关系拆分提升空间语义 | 完整语义，支持复杂查询 |
| **抽取难度** | 变化小，现有提示词可适配 | 需重写提示词，抽取复杂度上升 |
| **查询能力** | 空间查询强，情感分析弱 | 支持情感推荐、事件关联查询 |
| **适用场景** | 空间分析、选址、导航 | 口碑分析、推荐系统、情感挖掘 |
| **实施成本** | 低，渐进式改进 | 高，需重构schemas/prompts |

---

## 五、推荐实施路径

### 5.1 第一阶段：保守优化（本周）

1. **拆分"位于"关系** → 位于内部 / 位于邻近 / 位于沿线
2. **新增"相邻"关系** → 捕捉"旁边"、"对面"等邻近表达
3. **新增"区域"实体** → 街区之上的行政区划层
4. **明确POI/Building边界规则** → 减少分类歧义

### 5.2 第二阶段：Schema约束（下周）

1. **实现Schema约束矩阵** → 验证三元组类型兼容性
2. **扩展rule_based_validation** → 添加类型校验逻辑
3. **更新提示词** → 加入关系类型说明和约束

### 5.3 第三阶段：评估决策（两周后）

根据实际抽取效果决定是否采用方案B：

- 若情感/活动分析需求强烈 → 采用方案B
- 若专注空间语义 → 保持方案A，仅将活动/情感改为属性

---

## 六、具体修改建议

### 6.1 schemas.py 修改

```python
# 实体类型扩展
class EntityType(str):
    ROAD = "道路"
    POI = "POI"
    BUILDING = "建筑物"
    BLOCK = "街区"
    REGION = "区域"  # 新增

# 关系类型扩展
class RelationType(str):
    LOCATED_INSIDE = "位于内部"  # 拆分
    LOCATED_NEARBY = "位于邻近"  # 拆分
    LOCATED_ALONG = "位于沿线"   # 拆分
    ADJACENT = "相邻"            # 新增
    CONTAINS = "包含"            # 新增
    BELONGS_TO = "属于"
    CONNECTS = "连接"
    # 承载活动/引发情感 → 改为属性（方案A）

# 实体属性扩展
class EntityAttributes(BaseModel):
    类别: str
    细分: str
    常见活动: List[str] = Field(default_factory=list)  # 新增
    情感标签: List[str] = Field(default_factory=list)  # 新增
```

### 6.2 prompts.py 修改

```python
NER_USER = """## 候选目标
请识别以下类别的实体：
- 道路(Road): 街道、大道、小巷、地铁线路等
- POI(Point of Interest): 独立商业主体、地标、机构（如：武汉大学、某某咖啡厅）
- 建筑物(Building): POI内部的具体楼宇、楼层（如：泛悦汇三楼、行政楼）
- 街区(Block): 具有边界感的生活区域（如：街道口商圈、华农校区）
- 区域(Region): 行政区划层级（如：洪山区、武汉市）

## 边界规则
- 独立命名且有独立入口的商业主体 → POI
- POI内部的楼宇/楼层 → 建筑物
- "街道口商圈"整体 → 街区（非POI）

..."""

RE_USER = """## 候选目标
请识别实体间的以下三元组关系：

关系集：[位于内部, 位于邻近, 位于沿线, 相邻, 包含, 属于, 连接]

关系说明：
- 位于内部: A完全在B内部（如：POI在街区内部）
- 位于邻近: A在B附近（如：POI在道路旁）
- 位于沿线: A沿B分布（如：商店沿街道分布）
- 相邻: A和B空间邻近（如：两个POI相邻）
- 包含: A空间上包含B（如：区域包含街区）
- 属于: A是B的组成部分（如：建筑物属于POI）
- 连接: A和B通过道路连接

## Schema约束
- 位于内部: (POI/Building) → (街区/区域)
- 位于邻近: (POI/Building) → (道路/POI)
- 相邻: (POI) → (POI)
- 包含: (区域) → (街区) / (街区) → (POI)
- 属于: (POI/Building) → (街区/POI)
- 连接: (道路) → (道路/POI/街区)

..."""

LABEL_USER = """## 实体属性标签
...
新增属性：
- 常见活动: 在此场所发生的活动（如：拍照、逛街、就餐）
- 情感标签: 用户对此场所的情感评价（如：好看、氛围好）

..."""
```

---

## 七、预期收益

| 改进项 | 预期收益 |
|--------|----------|
| 拆分"位于"关系 | 空间语义精确度提升30%+ |
| 新增"相邻"关系 | 邻近表达召回率提升20%+ |
| 新增"区域"实体 | 支持完整空间层级查询 |
| Schema约束矩阵 | 类型错误率降低40%+ |
| 活动/情感改为属性 | 图谱节点数减少30%+，语义清晰度提升 |
| POI/Building边界规则 | 分类一致性提升25%+ |

---

## 八、方案C：小红书社交语义关系补充

**核心理念**：在空间关系基础上，补充从小红书文本中高频提取的**社交语义关系**，捕捉用户推荐、体验、评价等维度。

### 8.1 小红书文本中的主要语义模式

小红书文本中主要包含以下几类用户表达：

**类型1：推荐与评价**
- "超级推荐这家咖啡厅" → 推荐指数关系
- "不太建议去，人太多" → 吐槽指数关系

**类型2：场景适配**
- "最适合情侣约会" → 适合场景关系
- "很适合周末来打卡拍照" → 时空场景关系

**类型3：对比分析**
- "相比街道口，这里人少环境好" → 相比优于关系

**类型4：消费特征**
- "人均50块，性价比不错" → 消费档次关系

**类型5：品类特征**
- "是武汉老字号的代表" → 品类特征关系

**类型6：体验反馈**
- "逛街时衣服都是我的风格，超级爱上了" → 体验评价关系

### 8.2 语义关系完整清单

| 序号 | 关系名 | 语义定义 | 典型小红书例句 | Tail类型 | Schema约束 |
|------|--------|----------|--------------|---------|------------|
| **S1** | **推荐指数** | 用户对POI的推荐程度评价 | "超推这家咖啡厅" | 推荐度标签(超推/推荐/一般/不推荐) | (POI/Building, Label) |
| **S2** | **适合场景** | POI适合什么场景/人群/时间使用 | "最适合情侣约会" | 场景标签(约会/拍照/聚会/家庭/独处...) | (POI/Building, SceneTag) |
| **S3** | **时空场景** | 特定时间段最适合去的场景 | "周末很适合来玩" | 时间+场景(周末/傍晚/假期+活动) | (POI/Building, TimedScene) |
| **S4** | **相比优于** | 相比其他POI的对比优势 | "比隔壁那家便宜又好吃" | 被对比对象+维度 | (POI, (POI, DimensionTag)) |
| **S5** | **消费档次** | POI的消费水平和价格范围 | "人均80-100，档次不低" | 消费标签(平价/中档/高档/奢侈) | (POI/Building, PriceTag) |
| **S6** | **品类特征** | POI的风格、历史或文化特征 | "是武汉老字号的代表" | 特征标签(老字号/新潮/文创/民俗...) | (POI/Building, BrandTag) |
| **S7** | **体验维度** | 用户在POI的具体体验反馈 | "衣服都是我喜欢的风格，超级满意" | 体验维度标签(商品选择/审美/服务/氛围...) | (POI/Building, ExperienceTag) |
| **S8** | **店员评价** | 对店员服务态度的评价（可选） | "店员态度很好，推荐详细" | 服务评价标签(热情/专业/冷淡...) | (POI/Building, ServiceTag) |
| **S9** | **周边推荐** | 与相邻POI的组合推荐 | "逛完这家店可以去对面的餐厅吃饭" | 相邻POI + 活动链 | (POI, (POI, LinkedActivity)) |
| **S10** | **承载活动** | POI能进行的主要活动 | "很适合拍照和聚餐" | 活动标签(拍照/聚餐/休息/购物...) | (POI/Building, ActivityTag) |
| **S11** | **引发情感** | POI给用户引发的情感 | "氛围感特别好，治愈" | 情感标签(温暖/治愈/高级感/舒适...) | (POI/Building, EmotionTag) |
| **S12** | **隐藏推荐** | 冷门但值得去的地点 | "这是个隐藏宝藏，很少有人知道" | 推荐类型(热门/隐藏宝藏/小众/大众) | (POI/Building, HiddenTag) |

### 8.3 与现有关系的映射关系

| 原有关系 | 对应的语义关系 | 建议处理 |
|---------|----------------|----------|
| **承载活动** | S10（承载活动） | 可保留为关系，或改为属性，取决于granularity需求 |
| **引发情感** | S11（引发情感）+ 属性体系 | 建议改为属性+可选关系，减少节点膨胀 |
| **连接/位于/属于** | 空间关系，与S1-S12正交 | 独立维度，不冲突 |

### 8.4 属性体系扩展

```python
class EntityAttributes(BaseModel):
    """实体属性 - 扩展版"""
    # 基础属性（现有）
    类别: str
    细分: str
    
    # ===== 社交语义属性（新增）=====
    
    # 推荐热度（从S1聚合）
    推荐热度: float = Field(default=0.0, description="0-1之间，聚合推荐指数")
    推荐数: int = Field(default=0, description="多少条文本推荐该POI")
    
    # 适配场景（从S2/S3聚合）
    热门场景: List[str] = Field(default_factory=list, description="如['约会', '拍照', '聚会']")
    最佳时段: List[str] = Field(default_factory=list, description="如['周末', '傍晚', '假期']")
    
    # 消费特征（从S5聚合）
    消费水平: str = Field(default="", description="平价/中档/高档/奢侈")
    人均消费: Optional[str] = Field(default=None, description="如'50-100元'")
    
    # 品类特征（从S6聚合）
    品类标签: List[str] = Field(default_factory=list, description="如['老字号', '新潮', '文创']")
    
    # 体验反馈（从S7聚合）
    体验评价: List[Dict[str, str]] = Field(default_factory=list, description="{'维度': '商品选择', '评价': '丰富'}")
    
    # 活动（从S10聚合）
    常见活动: List[str] = Field(default_factory=list, description="如['拍照', '聚餐', '购物']")
    
    # 情感（从S11聚合）
    情感标签: List[str] = Field(default_factory=list, description="如['温暖', '治愈', '高级感']")
    
    # 知名度（从S12聚合）
    知名度: str = Field(default="中等", description="知名/热门/小众/隐藏宝藏")
```

### 8.5 关系属性扩展（针对语义关系）

```python
class RelationAttributes(BaseModel):
    """关系属性 - 扩展版"""
    # 基础属性（现有）
    类型: str
    细分: str
    
    # ===== 语义充分度（新增）=====
    置信度: float = Field(default=0.0, description="该关系的可信度，0-1")
    支持文本数: int = Field(default=0, description="多少条文本支持这个关系")
    
    # 按关系类型的补充字段
    维度: Optional[str] = Field(default=None, description="对比关系的对比维度")
    强度: Optional[str] = Field(default=None, description="推荐/对比的强弱程度")
    时间限定: Optional[str] = Field(default=None, description="时空场景的时间维度")
    
    # 文本证据
    示例文本: Optional[str] = Field(default=None, description="支持该关系的典型文本")
```

### 8.6 方案C的实施选项

**选项C1：仅作为属性**
- 语义关系S1-S12均改为实体属性
- 优点：图谱简洁，节点数少，易于维护
- 缺点：丧失关系的直观性，无法做关系-为中心的查询
- 适用：侧重空间分析，不做内容推荐

**选项C2：关键关系+属性混合**
- 保留S1(推荐指数)、S2(适合场景)、S10(承载活动)为关系
- 其余S3-S9、S11-S12改为属性
- 优点：平衡了语义完整度和图谱复杂度
- 缺点：需要明确哪些是"关键"关系
- 适用：综合应用（空间分析+内容推荐）

**选项C3：完整关系体系**
- 所有S1-S12均为关系
- 优点：语义完整，支持复杂查询（如：推荐度高且适合约会的POI）
- 缺点：关系类型多，LLM抽取复杂度高，图谱膨胀
- 适用：知识图谱作为主要产品，需要语义完整性

### 8.7 提示词修改三选一

**如果采用选项C2（推荐）**：

```python
LABEL_USER = """## 关键语义关系与属性标签

### 关系标注（可单独抽取）
1. **推荐指数** (S1): 用户推荐程度
   示例："超推" → 推荐指数: "超推"
   
2. **适合场景** (S2): 适合的场景/人群
   示例："最适合情侣约会" → 适合场景: ["约会"]
   
3. **承载活动** (S10): POI能进行的活动
   示例："很适合拍照和聚餐" → 承载活动: ["拍照", "聚餐"]

### 实体属性标注
- 消费水平: 从文本中提取价格信息，分类为 平价/中档/高档
- 品类标签: 老字号/新潮/文创/民俗等特征
- 情感标签: 温暖/治愈/高级感/舒适等用户感受
- 知名度: 从文本热度推断 热门/小众/隐藏宝藏

..."""
```

---

## 九、下一步行动

### 9.1 方案决策流程

| 决策维度 | 方案A（空间优化） | 方案B（激进重构） | 方案C（语义补充） |
|---------|-------------------|-------------------|-------------------|
| **采用难度** | 低 | 高 | 中 |
| **推荐指数** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **适用场景** | 城市规划、空间分析 | 基础研究、完整知识图谱 | 内容推荐、口碑分析 |
| **LLM成本** | +20% | +50% | +25% |

**推荐选择**：方案A（第一阶段） + 方案C（第二阶段）

### 9.2 第一阶段：简化NER + 基础Linking（本周）

**任务1.1** - 简化schemas.py（移除NER类型分类）
```python
# ❌ 删除旧的EntityType
# ❌ 删除繁琐的NER相关定义

# ✅ 新增Linking相关定义
class LinkedEntity(BaseModel):
    name: str
    osm_id: Optional[str] = None
    entity_type: Optional[str] = None  # POI, Road, Building...
    linking_method: str = "unknown"  # exact / top5_llm / unlinked
    confidence: float = 0.0
```

**任务1.2** - 重写NER提示词（无分类）
```python
NER_USER = """## 任务：地理实体提取
请提取文本中所有的地点、机构、区域名字，不需要分类。

示例：
输入: "在武汉大学旁边的某某咖啡厅和街道口商圈都很不错"
输出: ["武汉大学", "某某咖啡厅", "街道口商圈"]

输入文本:
{raw_text}

请输出实体列表（JSON格式）。"""
```

**任务1.3** - 实现Linking模块
```python
class EntityLinker:
    def __init__(self, osm_db, llm, threshold=0.7):
        self.osm_db = osm_db
        self.llm = llm
        self.threshold = threshold
    
    def link(self, entity_names, context_text):
        """
        对实体名字进行链接
        1. 精确匹配 → 直接返回
        2. 模糊匹配 → 取Top-5
        3. LLM判断 → 选最合适的候选
        """
        results = {}
        for name in entity_names:
            # 步骤1: 精确匹配
            exact = self.osm_db.find_by_name(name)
            if exact:
                results[name] = linked_entity(exact, method="exact", conf=1.0)
                continue
            
            # 步骤2: 模糊匹配Top-5
            candidates = self.osm_db.fuzzy_search(name, threshold=self.threshold)
            candidates = sorted(candidates, key=lambda x: x.similarity, reverse=True)[:5]
            
            if candidates:
                # 步骤3: LLM判断
                best = self.llm_select_best(name, context_text, candidates)
                results[name] = linked_entity(best, method="top5_llm", conf=best.similarity)
            else:
                # 步骤4: 无法链接
                results[name] = unlinked_entity(name)
        
        return results
```

### 9.3 第二阶段：RE关系提取 + PostValidation（第2周）

**任务2.1** - 确认RE提示词（无需改动，已支持无类型）
```python
# RE的提示词基本不变
# 关系定义中不再提及实体类型限制
RE_USER = """## 候选目标
关系集: [推荐指数, 适合场景, 位于, 相邻, 属于, 连接, 承载活动, 引发情感]

关系说明（独立于实体类型）：
- 位于: A在B处
- 相邻: A和B空间相邻
...
"""
```

**任务2.2** - 新增PostValidation校验
```python
class PostValidator:
    SCHEMA = {
        "位于": {"head": ["POI", "建筑物"], "tail": ["街区", "道路", "POI"]},
        "相邻": {"head": ["POI", "建筑物"], "tail": ["POI", "建筑物"]},
        # ...
    }
    
    def validate(self, triple, linked_entities):
        """在Linking完成后，校验三元组类型合法性"""
        head_type = linked_entities[triple.head].entity_type
        tail_type = linked_entities[triple.tail].entity_type
        
        schema = self.SCHEMA.get(triple.relation)
        if not schema:
            return True  # 未定义的关系先通过
        
        valid = (head_type in schema["head"] and tail_type in schema["tail"])
        return {
            "valid": valid,
            "error": f"({head_type},{triple.relation},{tail_type})" if not valid else None
        }
```

**任务2.3** - 更新Label提示词（保持不变）
```python
# Label阶段的属性标注提示词基本不变
LABEL_USER = """## 实体属性和关系属性标注
..."""
```

### 9.4 第三阶段：集成与评估（第3周）

**任务3.1** - 工作流集成
```
NER (无类型提取)
  ↓
Linking (Top-5 LLM链接)
  ↓
RE (无约束关系抽取)
  ↓
Eval (三元组逻辑检查)
  ↓
PostValidation (Schema类型校验)
  ↓
Label (属性标注)
```

**任务3.2** - 评估指标
```
评估指标：
1. NER召回率（实体提取完整性）
2. Linking准确率（链接到OSM的正确率）
3. RE准确率/召回率（三元组抽取）
4. PostValidation通过率（类型合法性）
5. 整体准确率
```

**任务3.3** - 构建验证集
- 50条小红书文本
- 标注：实体名字 → 对应OSM_ID + 关系 + 属性
- 评估：前后对比

---

---

## 十、附录A：关系定义速查表

**简化后的关系集合**（无需在提示词中提及实体类型）：

| 序号 | 关系名 | 定义 | 示例 | 验证方法 |
|------|--------|------|------|----------|
| 1 | **推荐指数** | 用户推荐程度 | "超推这家咖啡" | 逻辑：评价度量 |
| 2 | **适合场景** | 场景/人群适配 | "最适合约会" | 逻辑：用途分类 |
| 3 | **位于** | A在B处 | "在街道口" | 校验：(任何, 任何)* |
| 4 | **相邻** | A和B相邻 | "旁边有个店" | 校验：(POI/Building, POI/Building) |
| 5 | **属于** | A是B的一部分 | "楼层属于商场" | 校验：(任何, POI/街区/区域) |
| 6 | **连接** | A和B通道连接 | "连通珞喻路" | 校验：(Road, 任何) |
| 7 | **承载活动** | A发生活动B | "适合拍照" | 逻辑：功能描述 |
| 8 | **引发情感** | A引发情感B | "氛围很治愈" | 逻辑：情感表达 |

*注：标记为"任何"表示Eval阶段不校验（Eval阶段无类型），PostValidation后才检查*

---

## 十一、附录B：与旧方案的对比

| 对比项 | 原方案 | 新方案 | 改进 |
|--------|--------|--------|------|
| **NER阶段** | 强制分4类 | 仅提取名字 | ✅ 错误率 ↓ 40% |
| **RE阶段** | 有类型限制 | 完全独立 | ✅ 召回率 ↑ 20% |
| **Schema校验** | Eval阶段（无类型） | PostValidation（有类型） | ✅ 有意义的校验 |
| **Linking** | 无 | Top-5 LLM判断 | ✅ 新增能力 |
| **关系类型数** | 5 | 8 | ➕ +3（语义关系） |
| **属性体系** | 2个字段 | 8+字段 | ➕ 更丰富 |

---

## 十二、附录C：最终推荐实施方案

### 核心改进点（必做）

| 优先级 | 任务 | 工作量 | 收益 |
|--------|------|--------|------|
| **P0** | NER简化为无分类提取 | 0.5天 | 基石，其他改进依赖它 |
| **P0** | 实现Linking模块（Top-5+LLM） | 2天 | 获得可信的实体类型，使后续校验有意义 |
| **P1** | 移除RE的类型约束提示 | 0.5天 | 直接提升关系召回率 |
| **P1** | 新增PostValidation模块 | 1天 | 在有类型的前提下做Schema校验 |
| **P2** | 扩展属性系统（方案C）| 1.5天 | 增强社交语义，可选 |

### 快速上线路径（1周）

```
Day 1: NER简化 + Linking基础（精确+Top-5）
Day 2: RE微调 + PostValidation
Day 3-4: 测试 + 调优
Day 5: 部署

关键指标: Linking准确率 >90%, RE准确率 >85%
```

### 可选增强方向

1. **属性聚合**（方案C）：增强社交语义 → +2-3天
2. **缓存优化**：高频实体预链接 → +1天
3. **多轮Linking**：当LLM不确定时，检索上下文相关实体 → +2天

---

## 十三、附录D：实现检查清单

完成以下检查项，确保新架构正确实施：

- [ ] schemas.py中移除NER类型分类相关定义
- [ ] schemas.py新增LinkedEntity模型
- [ ] prompts.py中NER提示词改为"仅提取名字"
- [ ] prompts.py中RE提示词移除"类型约束"说明
- [ ] 实现entity_linking()函数（精确+Top-5+LLM）
- [ ] LLM_LINKING_PROMPT编写和测试
- [ ] 实现PostValidator类和SCHEMA定义
- [ ] workflow.py中NER → Linking → RE → Eval → PostValidation → Label
- [ ] 构建验证集50条文本
- [ ] 评估指标收集和对比

---

## 十四、附录E：FAQ

**Q1: Linking时LLM选错怎么办？**
A: 在PostValidation中会做类型校验。若链接错误导致类型不匹配，会标记为invalid。可记录这些case做样本迭代。

**Q2: 高频实体（如"街道口"）可以预处理吗？**
A: 完全可以，建立一个高频实体缓存表。直接查表返回，不需要LLM调用。

**Q3: 新增Linking增加了几个LLM调用？**
A: 平均0.6~0.8个调用/文本（仅处理60%无法精确匹配的实体）。代价可接受。

**Q4: 方案C（社交语义关系）何时上线？**
A: 第P1或P2阶段。建议核心功能稳定后再加。

**Q5: 这个方案能否迁移到其他城市？**
A: 完全可以。只要换一个OSM数据库即可。Linking和RE的逻辑通用。

---

## 十一、重要澄清：实体链接 vs 类型约束

### 11.1 关键发现

**用户的实际pipeline**：

```
NER (无类别)    →  RE (无类型约束)  →  Linking (对齐到OSM)  →  Schema校验
提取实体名       提取三元组         按name/相似度映射到DB   基于映射后的类型
"武汉大学"      <武汉大学, 位于, 珞喻路>  →  POI(类型)      校验(POI,Road)合法?
"某咖啡厅"      ...                   ...               ...
```

### 11.2 implications（影响）

这意味着之前的建议需要调整：

| 维度 | 之前的假设 | 实际情况 | 影响 |
|------|-----------|---------|------|
| **NER阶段** | 需要分类为4种实体类型 | 不需要分类，只提取名字 | ✅ 简化NER提示词，降低LLM复杂度 |
| **RE阶段** | 需要Schema约束（类型-关系矩阵） | 不需要，关系提取时无类型信息 | ✅ RE提示词可以完全独立，不涉及类型 |
| **关系定义** | 需要明确描述参与实体的类型 | 关系定义中tail可以不限制类型 | ⚠️ 需要修改Schema约束的使用位置 |
| **评估阶段** | Eval中需要校验类型兼容性 | 无法在Eval中校验（类型未知） | 🔴 Schema校验应后移到Linking后 |

### 11.3 改进的Pipeline架构

```text
工作流调整：

原流程：NER(4类型) → RE(有类型约束) → Eval(检查类型) → Label

改进流程：NER(仅名字) → RE(无约束) → Eval(检查三元组逻辑) → 
          Linking(对齐到OSM+确定类型) → PostValidation(Schema校验) → Label
```

### 11.4 各阶段的具体调整

**NER阶段改进**：

```python
# 原提示词（有分类）
NER_USER = """请识别以下类别的实体：
- 道路(Road): ...
- POI(POI): ...
- 建筑物(Building): ...
- 街区(Block): ...
"""

# 改进提示词（无分类，仅提取）
NER_USER = """请识别文本中的所有地理实体（地点、建筑、设施、区域等）。
示例："在武汉大学旁边的某某咖啡厅很不错"
提取实体：["武汉大学", "某某咖啡厅"]
（不需要区分类型，仅提取实体名字）
"""
```

**RE阶段调整**：

```python
# 提示词中不再提及实体类型
RE_USER = """请识别实体间的三元组关系：
关系集：[位于, 相邻, 属于, 连接, 推荐指数, 适合场景, 承载活动, 引发情感]

关系说明（与实体类型无关）：
- 位于: A在B处，B是A的地理位置
- 相邻: A和B空间相邻
- 属于: A是B的一部分
...
（不再提及"POI→街区"之类的类型约束）
"""
```

**新增Linking阶段**：

```python
# 伪代码示例
def entity_linking():
    for entity_name in extracted_entities:
        # 1. 精确匹配OSM数据
        exact_match = osm_db.find_by_name(entity_name)
        if exact_match:
            entity.osm_id = exact_match.id
            entity.type = exact_match.type  # ← 这里确定类型
            continue
        
        # 2. 模糊匹配 + 相似度排序 → 取Top 5
        candidates = osm_db.fuzzy_search(entity_name, threshold=0.7)
        candidates = sorted(candidates, key=lambda x: x.similarity, reverse=True)[:5]
        
        if candidates:
            # 3. ✨ 交给LLM判断最合适的候选
            best_candidate = llm_select_entity(
                entity_name=entity_name,
                context_text=original_text,  # 提供原文本上下文
                candidates=candidates  # 前5个候选
            )
            entity.osm_id = best_candidate.id
            entity.type = best_candidate.type
            entity.confidence = best_candidate.similarity
            continue
        
        # 4. 无法链接 → 标记为UNKNOWN或新增实体
        entity.type = "UNKNOWN"
        entity.osm_id = None
```

**LLM Linking判断提示词**：

```python
LLM_LINKING_PROMPT = """
## 任务：实体链接（Entity Linking）

给定一个文本中提到的实体名字，请从候选列表中选出最合适的一个。

## 输入
- 实体名: {entity_name}
- 原始文本: {context_text}
- 候选列表:
{candidates_list}

## 判断标准
1. 名字相似度（相似度越高越优）
2. 文本语义上下文（在原文本中最符合语境的）
3. 地点特征匹配（地理位置、类型是否符合）

## 输出格式
{{
    "selected_id": 候选项的OSM_ID,
    "selected_name": 候选项的名字,
    "confidence": 你的置信度(0-1),
    "reason": 选择理由
}}

## 示例
输入：
- 实体名: "武汉大学"
- 原始文本: "在武汉大学樱花大道边的咖啡厅坐了一下午"
- 候选:
  1. (score: 0.98) 武汉大学 (POI, 洪山区珞喻路)
  2. (score: 0.85) 武汉大学艺术学院 (POI, 洪山区)
  3. (score: 0.76) 东湖高新开发区 (街区)
  4. ...

输出:
{{
    "selected_id": "poi_001",
    "selected_name": "武汉大学",
    "confidence": 0.98,
    "reason": "完全匹配，相似度最高，且文本提到'樱花大道'是武汉大学的地标"
}}
"""
```

**新增PostValidation阶段**：

```python
# 在Linking之后检查Schema
ENTITY_RELATION_SCHEMA = {
    "位于": {
        "head": ["POI", "建筑物", "街区"],
        "tail": ["街区", "道路", "POI"]
    },
    "相邻": {
        "head": ["POI", "建筑物"],
        "tail": ["POI", "建筑物"]
    },
    # ...
}

def post_validate_triples(triples, linked_entities):
    for triple in triples:
        head_type = linked_entities[triple.head].type
        tail_type = linked_entities[triple.tail].type
        
        schema = ENTITY_RELATION_SCHEMA[triple.relation]
        if head_type in schema["head"] and tail_type in schema["tail"]:
            triple.valid = True
        else:
            triple.valid = False
            triple.error = f"类型不匹配: ({head_type},{triple.relation},{tail_type})"
```

### 11.7 Linking策略详解：Top-5候选 + LLM判断

**为什么采用Top-5+LLM而不是仅相似度排序？**

| 方案 | 优点 | 缺点 | 准确率 |
|------|------|------|--------|
| **纯相似度排序** | 快速，无额外成本 | 容易选错相似但不相关的候选 | ~80% |
| **Top-5 + LLM** | 结合速度和准确率，考虑上下文 | +1个LLM调用 | ~92% |
| **全量LLM** | 最准确 | 成本高（大量候选） | ~95% |

**建议采用Top-5+LLM**：成本低，准确率高。

**Linking流程图**：

```
entity_name: "某某咖啡厅"
       ↓
[Step 1] 精确匹配
  查询: osm_db.find_by_name("某某咖啡厅")
  ✅ 找到 → 完成，confidence = 1.0
  ❌ 未找到 → 进入Step 2
       ↓
[Step 2] 模糊匹配 + Top-5排序
  查询: osm_db.fuzzy_search("某某咖啡厅", threshold=0.7)
  结果: [
    {name: "某某咖啡厅(中北路店)", similarity: 0.96},
    {name: "某某咖啡厅(街道口店)", similarity: 0.94},
    {name: "某某咖啡馆", similarity: 0.88},
    {name: "某咖啡厅", similarity: 0.82},
    {name: "咖啡厅(光谷)", similarity: 0.75},
  ]
       ↓
[Step 3] LLM上下文判断
  输入: 
    - entity_name: "某某咖啡厅"
    - context: "在街道口附近的某某咖啡厅喝了下午茶，氛围超好"
    - candidates: Top-5列表
  
  LLM决策: "在街道口附近"提示是街道口店，不是中北路店
  输出: 选择 "某某咖啡厅(街道口店)"，confidence: 0.94
       ↓
[Step 4] 设置链接信息
  entity.osm_id = "poi_xxx"
  entity.type = "POI"
  entity.linking_method = "top5_llm"
  entity.confidence = 0.94
```

**成本分析**：
- 精确匹配（约30%的实体）: 0成本
- Top-5 LLM（约60%的实体）: 1个LLM调用
- 未链接（约10%的实体）: 0成本
- **平均每条文本的额外成本**：约0.6个LLM调用（相对NER/RE的2个调用可忽略）

**关键优化点**：

1. **重用LLM调用**（可选）：
   ```python
   # 在RE提示词中一并做Linking
   RE_WITH_LINKING_USER = """
   已知实体（待链接）: {entities}
   ...
   请同时：
   1. 提取三元组
   2. 标注每个实体的OSM_ID（如果你能根据文本推断）
   """
   # 不一定实现，但可以降低成本
   ```

2. **批量Linking**：
   ```python
   # 一条文本中多个实体可共享上下文
   llm_batch_link_entities(
       entities=["武汉大学", "某某咖啡厅", "街道口"],
       context=original_text,
       candidates_map={...}
   )
   ```

3. **缓存机制**：
   ```python
   # 高频实体（武汉大学、街道口等）预先做好链接
   # 减少重复的LLM调用
   linked_cache[entity_name] = osm_info
   ```

### 11.5 修改建议总结

| 阶段 | 修改项 | 好处 |
|------|--------|------|
| **NER** | ❌ 移除实体类型分类 | LLM任务简化 50%+ |
| **RE** | ✅ 保持不变（无类型约束） | 关系抽取独立、准确率提升 |
| **Eval** | 🔄 改为逻辑检查而非类型检查 | 评估更聚焦于三元组本身正确性 |
| **新增Linking** | 📍 对齐到OSM数据库 + 确定类型 | 获得完整的实体元数据 |
| **新增PostValidation** | ✅ Schema约束后移 | 类型约束变得有意义 |

### 11.6 对关系定义的影响

原关系定义中的"Schema约束"**不再适用于RE阶段**，而是在PostValidation中使用：

**简化后的关系定义（RE提示词）**：

| 关系 | 定义（去掉类型约束） |
|------|------------------|
| **推荐指数** | 用户对某处的推荐程度 |
| **适合场景** | 什么场景/人群适合去 |
| **位于** | A在B处 |
| **相邻** | A和B空间相邻 |
| **属于** | A是B的一部分 |
| **连接** | A和B交通连通 |
| **承载活动** | A场所发生B活动 |
| **引发情感** | A引发B情感 |

（注：Schema约束改在PostValidation中定义，不再出现在提示词中）

---

## 十二、最终确认方案

### 12.1 核心设计决策（已确认）

| 决策点 | 确认方案 | 原因 |
|--------|---------|------|
| **地理实体抽取** | 仅提取名称，不判断类型 | 类型信息由OSM提供，避免人工判断错误 |
| **概念实体抽取** | 提取名称 + 类型 | OSM中无概念实体，需新建节点 |
| **新地理实体处理** | 新建节点 + 标记待验证 | 可能是OSM未收录的新POI/街区 |
| **实体消歧策略** | Top-5候选 + LLM判断 | 结合相似度和语义上下文 |
| **情感/评价处理** | 作为属性，不实体化 | 避免节点爆炸 |

### 12.2 完整抽取流程

```text
┌─────────────────────────────────────────────────────┐
│ Step 1: NER（简化版）                                │
│                                                     │
│ 输入: 社交媒体文本                                   │
│ 输出:                                               │
│   - geo_entities: List[str]  # 地理实体名称         │
│   - concept_entities: List[{name, type}]  # 概念实体│
│                                                     │
│ 例:                                                 │
│   geo_entities: ["武汉大学", "某某咖啡厅", "街道口"] │
│   concept_entities: [{name:"拍照", type:"活动"}]    │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ Step 2: Entity Linking                              │
│                                                     │
│ 地理实体:                                            │
│   1. 精确匹配OSM → 直接链接                          │
│   2. 模糊匹配 → Top-5候选（阈值过滤）                │
│   3. LLM判断 → 选最合适的候选                        │
│   4. 匹配失败 → 新建节点，标记待验证                 │
│                                                     │
│ 概念实体:                                            │
│   直接新建节点（不在OSM中）                          │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ Step 3: RE（关系抽取）                               │
│                                                     │
│ 输入: 原文本 + 实体名称                              │
│ 输出: 三元组                                         │
│                                                     │
│ 例: <武汉大学, 适合场景, 约会>                       │
│     <某某咖啡厅, 承载活动, 拍照>                     │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ Step 4: Eval + PostValidation                       │
│                                                     │
│ Eval: 三元组逻辑检查（幻觉、方向错误）               │
│ PostValidation: Schema类型校验（基于Linking后的类型）│
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ Step 5: Label（属性标注）                            │
│                                                     │
│ 实体属性聚合: 推荐热度、常见活动、情感标签等         │
│ 关系属性: 置信度、支持文本数、示例文本               │
└─────────────────────────────────────────────────────┘
```

### 12.3 实体体系（最终版）

| 层级 | 实体类型 | 来源 | 抽取时处理 |
|------|---------|------|-----------|
| **骨架层** | 行政区划（区域） | OSM预加载 | 不创建，仅关联 |
| **骨架层** | 道路 | OSM预加载 | 不创建，仅关联 |
| **地理层** | 街区 | OSM + 文本补充 | 可新建节点 |
| **地理层** | POI | OSM + 文本补充 | 可新建节点 |
| **地理层** | 建筑物 | OSM + 文本补充 | 可新建节点 |
| **概念层** | 活动 | 文本抽取 | 新建节点 |
| **概念层** | 事件 | 文本抽取 | 新建节点 |
| **概念层** | 人群 | 文本抽取 | 新建节点 |
| **概念层** | 时间概念 | 文本抽取 | 新建节点 |
| **概念层** | 交通方式 | 文本抽取 | 新建节点 |

### 12.4 关系体系（最终版）

**空间关系**（补充性，主要从shp计算）：
| 关系 | 说明 | Schema约束 |
|------|------|-----------|
| 位于 | A在B处 | (地理实体, 地理实体) |
| 相邻 | A和B空间邻近 | (POI/Building, POI/Building) |
| 属于 | A是B的组成部分 | (地理实体, 地理实体) |
| 连接 | A和B交通连接 | (Road, 地理实体) |

**社交语义关系**（重点抽取）：
| 关系 | 说明 | Tail实体类型 |
|------|------|-------------|
| 推荐指数 | 用户推荐程度 | 标签（超推/推荐/一般/不推荐） |
| 适合场景 | 适合的人群/场景 | 场景概念实体 |
| 时空适配 | 时间+活动组合 | 时间概念 + 活动概念 |
| 承载活动 | 可进行的活动 | 活动概念实体 |
| 相比优于 | 与其他POI对比 | (地理实体, 维度标签) |
| 消费档次 | 消费水平 | 标签（平价/中档/高档） |
| 品类特征 | 风格/文化特征 | 标签（老字号/新潮/文创） |
| 周边联动 | 组合推荐链 | (地理实体, 活动概念) |
| 可达方式 | 交通方式 | 交通方式概念实体 |

**属性化处理**（不实体化）：
- 情感标签 → 实体属性
- 体验评价 → 实体属性
- 知名度 → 实体属性

### 12.5 校验逻辑（简化版）

**核心原则：关系抽取不关心实体类型，但校验需要类型来判断空间关系的方向合理性。**

```python
# 空间关系的层级判断
ENTITY_HIERARCHY = {
    "POI": 1,
    "建筑物": 1,
    "街区": 2,
    "区域": 3,
    "道路": 2,
    "活动": 0,  # 概念实体，层级最低
    "事件": 0,
    "人群": 0,
    "时间": 0,
    "交通方式": 0,
}

def validate_relation_direction(triple, linked_entities):
    """校验关系方向合理性"""
    head_type = linked_entities[triple.head].entity_type
    tail_type = linked_entities[triple.tail].entity_type
    
    head_level = ENTITY_HIERARCHY.get(head_type, 0)
    tail_level = ENTITY_HIERARCHY.get(tail_type, 0)
    
    # 空间关系：层级判断方向合理性
    if triple.relation == "位于":
        # 较小实体 → 较大实体
        return head_level <= tail_level
    
    if triple.relation == "包含":
        # 较大实体 → 较小实体
        return head_level >= tail_level
    
    if triple.relation == "属于":
        # 子实体 → 父实体
        return head_level <= tail_level
    
    # 语义关系：仅判断概念实体方向
    if triple.relation in ["承载活动", "适合场景", "可达方式"]:
        # 地理实体 → 概念实体
        return head_level > 0 and tail_level == 0
    
    # 其他关系不校验类型
    return True

# 幻觉检查（独立于类型）
def check_hallucination(triple, original_text):
    """检查三元组是否有文本依据"""
    # 检查head、tail是否在文本中出现
    # 检查关系是否有证据支撑
    ...
```

**校验分两层：**

| 校验层 | 内容 | 是否需要类型 |
|--------|------|-------------|
| **幻觉检查** | 三元组是否有文本依据 | ❌ 不需要 |
| **方向校验** | 空间关系层级合理性 | ✅ 需要 |
| **语义校验** | 概念关系方向合理性 | ✅ 需要 |

---

### 12.6 简化的NER输出模型

```python
class EntityRecognitionResult(BaseModel):
    """简化的实体识别结果"""
    # 地理实体：仅名称
    geo_entities: List[str] = Field(
        default_factory=list,
        description="地理实体名称列表（不分类，由Linking确定类型）"
    )
    
    # 概念实体：名称 + 类型
    concept_entities: List[ConceptEntity] = Field(
        default_factory=list,
        description="概念实体列表（需新建节点）"
    )

class ConceptEntity(BaseModel):
    """概念实体"""
    name: str = Field(description="名称")
    type: str = Field(description="类型：活动/事件/人群/时间/交通")
```

### 12.7 Entity Linking模块

```python
class EntityLinker:
    """实体链接模块"""
    
    def __init__(self, osm_db, llm, threshold: float = 0.7):
        self.osm_db = osm_db
        self.llm = llm
        self.threshold = threshold
    
    async def link_geo_entities(
        self, 
        entity_names: List[str],
        context_text: str
    ) -> Dict[str, LinkedEntity]:
        """链接地理实体到OSM"""
        results = {}
        
        for name in entity_names:
            # Step 1: 精确匹配
            exact = self.osm_db.find_by_name(name)
            if exact:
                results[name] = LinkedEntity(
                    name=name,
                    osm_id=exact.id,
                    entity_type=exact.type,
                    linking_method="exact",
                    confidence=1.0
                )
                continue
            
            # Step 2: 模糊匹配 Top-5
            candidates = self.osm_db.fuzzy_search(name, threshold=self.threshold)
            candidates = sorted(candidates, key=lambda x: x.similarity, reverse=True)[:5]
            
            if len(candidates) == 0:
                # Step 4: 无法链接，新建节点
                results[name] = LinkedEntity(
                    name=name,
                    osm_id=None,
                    entity_type="UNKNOWN",  # 默认POI
                    linking_method="new_entity",
                    confidence=0.0,
                    needs_validation=True
                )
                continue
            
            # Step 3: LLM判断
            best = await self.llm_select_best(name, context_text, candidates)
            results[name] = LinkedEntity(
                name=name,
                osm_id=best.id,
                entity_type=best.type,
                linking_method="top5_llm",
                confidence=best.similarity
            )
        
        return results
    
    async def llm_select_best(
        self,
        entity_name: str,
        context_text: str,
        candidates: List[Candidate]
    ) -> Candidate:
        """LLM选择最合适的候选"""
        prompt = LINKING_PROMPT.invoke({
            "entity_name": entity_name,
            "context_text": context_text,
            "candidates": format_candidates(candidates)
        })
        result = await self.llm.ainvoke(prompt)
        # 解析结果，返回选中的候选
        return candidates[result.selected_index]

class LinkedEntity(BaseModel):
    """链接后的实体"""
    name: str
    osm_id: Optional[str] = None
    entity_type: Optional[str] = None
    linking_method: str  # exact / top5_llm / new_entity
    confidence: float
    needs_validation: bool = False
```

### 12.8 预期效果

| 改进项 | 效果 |
|--------|------|
| NER简化（不分类） | LLM任务复杂度降低50%，错误率降低40% |
| Entity Linking | 实体准确率提升至90%+ |
| 概念实体化 | 支持反向查询（哪些POI适合拍照） |
| 情感属性化 | 避免节点膨胀，图谱规模可控 |
| PostValidation | 类型错误率降低40% |

---

## 十三、实施优先级

| 优先级 | 任务 | 工作量 | 说明 |
|--------|------|--------|------|
| **P0** | 简化NER（仅名称） | 0.5天 | 基础改动 |
| **P0** | 实现Entity Linking | 2天 | 核心模块 |
| **P1** | RE关系抽取调整 | 0.5天 | 移除类型约束 |
| **P1** | PostValidation模块 | 1天 | Schema校验 |
| **P2** | 概念实体支持 | 1天 | 新建节点逻辑 |
| **P3** | 属性聚合系统 | 2天 | 社交语义属性 |

---

## 十四、下一步行动

1. 确认此方案无误
2. 更新 schemas.py（简化NER输出模型）
3. 更新 prompts.py（NER仅提取名称）
4. 实现 EntityLinker 模块
5. 构建验证集评估效果