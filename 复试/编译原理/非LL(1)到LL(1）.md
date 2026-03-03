### 提取公共左因子

$(A \to \alpha B \mid \alpha C) \to (A \to \alpha A' \space \space A \to B\mid C)$

### 消除左递归

#### 直接左递归
$(S \to S A \space \space S \to b  ) \to (S \to b S' \space \space S' \to A S' \mid \epsilon  )$

#### 间接左递归
将间接转为直接
$(S \to B A  \space \space B \to Sb \mid d) \to (S \to S b A \mid d A    )$ 
$\to(S\to dA A' \space \space A'\to bAA' \mid \epsilon)$


#### 二义性的消除
结构分层

E → E + E
E → E * E
E → (E)
E → id

改成

E → E + T | T
T → T * F | F
F → (E) | id


消除“模糊选择”

S → if E then S
S → if E then S else S
S → other

若句子为
If E 1 then if E 2 then S 1 else S 2
则分不清 else 属于那个 if


To

S → M | U

M → if E then M else M | other
U → if E then S | if E then M else U