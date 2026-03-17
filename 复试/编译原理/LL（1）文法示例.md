## 1. 定义文法 (Grammar)

假设我们要推导表达式 `i + i * i`。为了符合 LL(1) 属性（消除左递归和提取左公因子），文法如下：

1. $E \rightarrow T E'$
    
2. $E' \rightarrow + T E' \mid \epsilon$
    
3. $T \rightarrow F T'$
    
4. $T' \rightarrow * F T' \mid \epsilon$
    
5. $F \rightarrow ( E ) \mid i$
    

---

## 2. 预测分析表 (LL(1) Parsing Table)

在推导之前，编译器会根据 **First** 集和 **Follow** 集构建一张表，决定在当前非终结符下，看到哪个输入符该使用哪条规则：

|**非终结符**|**i**|**+**|**∗**|**(**|**)**|**$**|
|---|---|---|---|---|---|---|
|**$E$**|$E \rightarrow T E'$|||$E \rightarrow T E'$|||
|**$E'$**||$E' \rightarrow + T E'$|||$E' \rightarrow \epsilon$|$E' \rightarrow \epsilon$|
|**$T$**|$T \rightarrow F T'$|||$T \rightarrow F T'$|||
|**$T'$**||$T' \rightarrow \epsilon$|$T' \rightarrow * F T'$||$T' \rightarrow \epsilon$|$T' \rightarrow \epsilon$|
|**$F$**|$F \rightarrow i$|||$F \rightarrow ( E )$|||

---

## 3. 推导示例：输入 `i + i * i`

LL(1) 总是替换当前句型中最左边的非终结符。

|**步骤**|**当前句型 (Sentential Form)**|**选用的产生式**|**剩余输入**|
|---|---|---|---|
|1|**$E$**|$E \rightarrow T E'$|`i + i * i $`|
|2|**$T$** $E'$|$T \rightarrow F T'$|`i + i * i $`|
|3|**$F$** $T' E'$|$F \rightarrow i$|`i + i * i $`|
|4|$i$ **$T'$** $E'$|$T' \rightarrow \epsilon$|`+ i * i $`|
|5|$i$ **$E'$**|$E' \rightarrow + T E'$|`+ i * i $`|
|6|$i +$ **$T$** $E'$|$T \rightarrow F T'$|`i * i $`|
|7|$i +$ **$F$** $T' E'$|$F \rightarrow i$|`i * i $`|
|8|$i + i$ **$T'$** $E'$|$T' \rightarrow * F T'$|`* i $`|
|9|$i + i *$ **$F$** $T' E'$|$F \rightarrow i$|`i $`|
|10|$i + i * i$ **$T'$** $E'$|$T' \rightarrow \epsilon$|`$`|
|11|$i + i * i$ **$E'$**|$E' \rightarrow \epsilon$|`$`|
|12|$i + i * i$|**推导成功**||