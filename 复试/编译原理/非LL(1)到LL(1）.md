### 提取公共左因子

$(A \to \alpha B \mid \alpha C) \to (A \to \alpha A' \space \space A \to B\mid C)$

### 消除左递归

#### 直接左递归
$(S \to S A \space \space S \to b  ) \to (S \to b S' \space \space S' \to A S' \mid \epsilon  )$

#### 间接左递归
将间接转为直接
$(S \to B A  \space \space B \to Sb \mid d) \to (S \to S b A \mid d A    )$ 
$\to(S\to dA A' \space \space A'\to bAA' \mid \epsilon)$
