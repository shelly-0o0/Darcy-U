# 第十二阶段补充：学习细化动作的近邻工作与新颖性边界

核查日期：2026-11-20。本文件在第十二阶段实验协议冻结之后补充，只提供相关工作语境，不修改实验规则或通过门槛，也不据此启动训练。本次核查聚焦五篇直接学习网格细化决策的工作，不是完整或穷尽的新颖性检索。

**直接学习细化动作、用图网络传递邻域信息、平衡误差与资源、考虑多个误差分量以及细化的非局部影响，都已有明确先例。** 因此，即使本阶段精确动作 oracle 出现正信号，也只能说明当前离散问题的选择策略存在空间，不能把“从学伴随场改为学动作”本身写成新方法。

## 1. Deep Reinforcement Learning for Adaptive Mesh Refinement

作者：Corbin Foucart、Aaron Charous、Pierre F. J. Lermusiaux。预印本首次提交于 2022-09-25；正式发表于 *Journal of Computational Physics* 491（2023 年 10 月），文章号 112381，DOI `10.1016/j.jcp.2023.112381`。[作者预印本与日期](https://arxiv.org/abs/2209.12351)；[MIT 作者机构库的正式发表信息](https://dspace.mit.edu/entities/publication/f26aca21-c806-4375-8006-da6043b7d266)。

已完成的内容：把局部细化、粗化、不动作建模为部分可观测序贯决策；从数值模拟训练策略，不要求预先生成精确解数据集。第 2.3 节用动作前后数值解的变化构造精度奖励，并结合资源成本和预算障碍项。[方法正文](https://arxiv.org/html/2209.12351v1)。

与本项目的关系：以“动作产生的效果”代替单纯复现已有指标，并不是空白方向。其解变化奖励不能直接等同于本项目三个线性泛函的真实误差下降；本项目仍需用实际目标值验证，不能从能量或场变化推导目标误差必然改善。该差别是评价任务的区别，尚不构成新颖性证明。

## 2. Multi-Agent Reinforcement Learning for Adaptive Mesh Refinement

作者：Jiachen Yang、Ketan Mittal、Tarik Dzanic、Socratis Petrides、Brendan Keith、Brenden Petersen、Daniel Faissol、Robert Anderson。AAMAS 2023，会议日期 2023-05-29 至 2023-06-02；预印本首次提交于 2022-11-02。[正式会议论文](https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p14.pdf)；[预印本记录](https://arxiv.org/abs/2211.00801)。

已完成的内容：VDGN 将每个元素视为协作智能体，用图网络和价值分解学习并行细化／粗化，处理元素分裂与删除后的信用分配。其多目标扩展优化误差与成本的 Pareto 权衡；第 7.2 节还以受限动作空间中的穷举搜索作对照。[正式论文第 3–7 节](https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p14.pdf)。

与本项目的关系：联合决策、预算权衡和用离线搜索检查动作空间都已有先例。这里的“多目标”明确包括误差与成本，并不自动等同于本项目同一主解的三个线性泛函；同样，本项目的单次、固定新增自由度检验也不能借用其动态问题上的结果作为泛化或效率证据。

## 3. Swarm Reinforcement Learning for Adaptive Mesh Refinement

作者：Niklas Freymuth、Philipp Dahlinger、Tobias Würth、Simon Reisch、Luise Kärger、Gerhard Neumann。*Advances in Neural Information Processing Systems* 36，NeurIPS 2023；核查的作者版本为 arXiv v3，日期 2023-10-09。[正式会议记录](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e85454a113e8b41e017c81875ae68d47-Abstract-Conference.html)；[版本记录](https://arxiv.org/abs/2304.00818)。

已完成的内容：ASMR 用可分裂的协作元素智能体、局部空间奖励和消息传递网络，直接输出网格标记，并交给经典有限元求解器。它已研究静态网格上的多轮细化，包含 Poisson、Stokes、弹性和热扩散等任务；不是只预测解场或伴随误差。[正文第 1、3、4 节及附录 B](https://arxiv.org/html/2304.00818v3)。

与本项目的关系：“GNN 选择局部动作，再由 FEM 求解”不是可单独主张的架构创新。若未来进行学习方法比较，不能只与指标排序比较后宣称超过已有学习网格方法；第十二阶段目前没有训练或复现 ASMR，也没有建立相对它的性能优势。

## 4. Adaptive Swarm Mesh Refinement Using Deep Reinforcement Learning with Local Rewards

作者：Niklas Freymuth、Philipp Dahlinger、Tobias Würth、Simon Reisch、Luise Kärger、Gerhard Neumann。*Machine Learning* 115，文章 169；出版社标注正式发布日期为 **2026-07-03**，DOI `10.1007/s10994-026-07064-4`。日期以出版社正式记录为准。[正式全文与发表记录](https://link.springer.com/article/10.1007/s10994-026-07064-4)。

已完成的内容：ASMR++ 延伸前述 ASMR，用最大局部误差下降和新增元素惩罚构造奖励，将惩罚参数作为策略输入。第 2.4 节明确讨论多个 quantities of interest，对归一化解分量的奖励聚合；还明确指出椭圆 PDE 细化存在非局部影响，因此加入全局回报。第 4.3 节提供相关消融。[正式全文第 2.4–2.5、4.3 节](https://link.springer.com/article/10.1007/s10994-026-07064-4)。

与本项目的关系：不能声称已有方法都忽略耦合或只考虑单一量。其文中的分量／场误差聚合和元素惩罚，与本项目“三个有符号线性泛函误差的缩放二范数、精确新增 64 个独立自由度、Schur 联合动作评价”是不同的具体问题定义。**区别尚未证明构成方法贡献，也未证明该文不能适配本项目。**

## 5. Learning Robust Marking Policies for Adaptive Mesh Refinement

作者：Andrew Gillette、Brendan Keith、Socratis Petrides。*SIAM Journal on Scientific Computing* 46(1)，A264–A289；在线发表于 2024-01-24，DOI `10.1137/22M1510613`。作者预印本 v2 日期为 2023-12-26。[正式期刊记录](https://epubs.siam.org/doi/abs/10.1137/22M1510613)；[作者全文](https://arxiv.org/html/2207.06339v2)。

已完成的内容：在经典 AFEM 中插入决策环节，学习随迭代变化的标记参数，而非固定阈值；研究满足精度目标的成本最小化及给定成本预算的精度问题，并扩展至 h/hp 细化。实验保留经典误差估计器，学习如何使用它。[正文第 2、4、5 节](https://arxiv.org/html/2207.06339v2)。

与本项目的关系：第十一阶段“准确指标不一定对应好的预算选择”这一现象，与已有的标记策略研究直接相关。不能把“改进决策而非提高估计器精度”描述为首次发现。本项目下一步若取得提升，还须区分来自一般标记参数优化、具体动作族、目标误差信息还是联合耦合评价。

## 对本阶段结果可作出的解释

以下是本项目的研究判断，不是上述论文对本实验的结论：

| 本项目中的设计元素 | 当前能否独立作为新颖性主张 |
|---|---|
| 用学习器输出细化动作 | 不能；前述 RL-AMR 工作已直接学习决策。 |
| 用图消息传递表示邻域与协作 | 不能；VDGN 与 ASMR 已采用图结构策略。 |
| 同时考虑准确性和计算资源 | 不能；多篇工作已明确优化这一权衡。 |
| 考虑多个量与非局部作用 | 不能笼统声称首次；ASMR++ 明确讨论这些问题。 |
| 当前 Schur 公式与增量条件化 | 是经典块消元，不是本项目提出的新数学公式。 |
| 三线性泛函、精确新自由度预算及联合动作的具体组合 | 目前只是明确的实验设置；本次有限检索既没有证明其独创，也没有证明其已有方法不可解决。 |

第十二阶段的 oracle 依赖一次精确细主问题求解。若其误差优于指标排序，只能证明所检验候选族的某条搜索路径具有更好的误差结果；不证明可以在推理时无参考地取得同样收益，也不证明 AI 必要、全局最优、完整加速或新颖性。AMG 目标代理与 uncoupled 对照用于进一步拆分这些问题，不应因 oracle 结果较好就跳过其失败或计算开销。

## 核查范围

使用来源均为作者 arXiv、会议官方论文、期刊出版社或作者机构库。Foucart 等期刊站点此次返回访问限制，因此方法依据作者 arXiv v1，正式元数据依据 MIT 机构库；未假称读过受限期刊全文。ASMR 2023 的官方 PDF 因大小限制未直接解析，方法依据作者 arXiv v3 HTML；会议身份依据官方记录。其余使用可读的正式全文／作者版本与相应出版社记录。

本次没有复现这些学习方法，也没有系统检索全部目标导向 AMR、组合优化或网格策略工作。因此不作“无人研究”“第一个”或“已达到国际一区创新性”的断言。本文件也不改变[第十二阶段冻结协议](stage12_protocol_zh.md)中的候选方法和继续标准。
