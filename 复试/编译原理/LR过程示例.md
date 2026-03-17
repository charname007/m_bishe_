## 1. 扩充文法 (Augmented Grammar)

首先引入一个新的开始符号 $S'$，得到扩充文法 $G'$：

(0) $S' \to S$

(1) $S \to S S +$

(2) $S \to S S *$

(3) $S \to a$

---

## 2. 构造 LR(0) 项集族 ($C$)

我们需要从初始项开始，通过 $CLOSURE$ 和 $GOTO$ 操作计算所有状态。

- **$I_0$:**
    
    $S' \to \cdot S$
    
    $S \to \cdot S S +$
    
    $S \to \cdot S S *$
    
    $S \to \cdot a$
    
- **$I_1 = GOTO(I_0, S)$:**
    
    $S' \to S \cdot$
    
    $S \to S \cdot S +$
    
    $S \to S \cdot S *$
    
    (由于点后面还有 $S$，需加入 $S$ 的产生式)
    
    $S \to \cdot S S +$
    
    $S \to \cdot S S *$
    
    $S \to \cdot a$
    
- **$I_2 = GOTO(I_0, a)$:**
    
    $S \to a \cdot$ (归约态)
    
- **$I_3 = GOTO(I_1, S)$:**
    
    $S \to S S \cdot +$
    
    $S \to S S \cdot *$
    
    $S \to S \cdot S +$
    
    $S \to S \cdot S *$
    
    $S \to \cdot S S +$
    
    $S \to \cdot S S *$
    
    $S \to \cdot a$
    
- **$I_4 = GOTO(I_3, +)$:**
    
    $S \to S S + \cdot$ (归约态)
    
- **$I_5 = GOTO(I_3, *)$:**
    
    $S \to S S * \cdot$ (归约态)
    

---

## 3. 计算 FOLLOW 集

构造 SLR 表需要知道非终结符的 $FOLLOW$ 集。

- $FOLLOW(S') = \{ \$ \}$
    
- $FOLLOW(S)$：
    
    1. 根据 $S' \to S$，$\in FOLLOW(S)$。
        
    2. 根据 $S \to S S +$，第二个 $S$ 后面是 $+$，所以 $+\in FOLLOW(S)$。
        
    3. 根据 $S \to S S +$，第一个 $S$ 后面是 $S$，所以 $FIRST(S) \in FOLLOW(S)$。即 $a \in FOLLOW(S)$。
        
    4. 同理，从 $S \to S S *$ 得到 $*\in FOLLOW(S)$。
        
- **结论：** $FOLLOW(S) = \{ a, +, *, \$ \}$
## 语法分析表概览 (Action/Goto)

|**状态**|**a**|**+**|*****|**$**|**S**|
|---|---|---|---|---|---|
|**0**|s2||||1|
|**1**|s2|||acc|3|
|**2**|r3|r3|r3|r3||
|**3**|s2|s4|s5||3|
|**4**|r1|r1|r1|r1||
|**5**|r2|r2|r2|r2||




| **步骤** | **状态栈** | **符号栈** | **剩余输入** | **动作 (ACTION)**         | **GOTO**       |
| ------ | ------- | ------- | -------- | ----------------------- | -------------- |
| (1)    | 0       | #       | aa*a+$   | **s2** (移进 a)           |                |
| (2)    | 0 2     | # a     | a*a+$    | **r3** (归约 S $\to$ a)   | GOTO(0, S) = 1 |
| (3)    | 0 1     | # S     | a*a+$    | **s2** (移进 a)           |                |
| (4)    | 0 1 2   | # S a   | *a+$     | **r3** (归约 S $\to$ a)   | GOTO(1, S) = 3 |
| (5)    | 0 1 3   | # S S   | *a+$     | **s5** (移进 *)           |                |
| (6)    | 0 1 3 5 | # S S * | a+$      | **r2** (归约 S $\to$ SS*) | GOTO(0, S) = 1 |
| (7)    | 0 1     | # S     | a+$      | **s2** (移进 a)           |                |
| (8)    | 0 1 2   | # S a   | +$       | **r3** (归约 S $\to$ a)   | GOTO(1, S) = 3 |
| (9)    | 0 1 3   | # S S   | +$       | **s4** (移进 +)           |                |
| (10)   | 0 1 3 4 | # S S + | $        | **r1** (归约 S $\to$ SS+) | GOTO(0, S) = 1 |
| (11)   | 0 1     | # S     | $        | **acc** (接受)            |                |