**SDD 是“做什么”（规[[复试/离散数学/格|格]]说明），而 SDT 是“怎么做”（具体实现）**。

SDD它不规定计算这些属性的具体先后顺序，只要最后结果满足等式即可。

SDT 明确指出执行时机

若自底向上，则 SDT 通常只用综合属性，且语义动作全部放在产生式的**最右端**。
否则综合属性和继承属性合用



## SDD 示例（L-SDD）

| **产生式**                   | **语义规则**                                                |
| ------------------------- | ------------------------------------------------------- |
| 1) $L \to E \mathbf{n}$   | $L.val = E.val$                                         |
| 2) $E \to T E'$           | $E'.inh = T.val$<br>$E.val = E'.syn$                    |
| 3) $E' \to + T E'_1$      | $E'_1.inh = E'.inh + T.val$<br>$E'.syn = E'_1.syn$      |
| 4) $E' \to \epsilon$      | $E'.syn = E'.inh$                                       |
| 5) $T \to F T'$           | $T'.inh = F.val$<br>$T.val = T'.syn$                    |
| 6) $T' \to * F T'_1$      | $T'_1.inh = T'.inh \times F.val$<br>$T'.syn = T'_1.syn$ |
| 7) $T' \to \epsilon$      | $T'.syn = T'.inh$                                       |
| 8) $F \to ( E )$          | $F.val = E.val$                                         |
| 9) $F \to \mathbf{digit}$ | $F.val = \mathbf{digit}.lexval$                         |

对于 L-SDT ，例如 $S\to \{{a_{1}}\}A\{{a_{2}}\}B\{{a_{3}}\}C\{{a_{4}}\}$ 一般产生式前的为继承属性赋值，a 1->A, a 2->B, a 3->C, 而最后则是 S 的综合属性赋值 a 4