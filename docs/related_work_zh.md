# 相关工作核对与本项目的创新边界

核对日期：2026-11-19。以下来自论文作者的 arXiv 页面、全文或出版社页面。它是聚焦的最近邻核查，不是完整系统综述；不能据此宣布“首次”。预印本年份与期刊出版年份分别标注，未核实的期刊信息不填写。

## 已确认的文献

### 1. 传统序列问题的粗空间复用

D. Kalchev, C. Ketelsen, P. S. Vassilevski, *Two-Level Adaptive Algebraic Multigrid for a Sequence of Problems with Slowly Varying Random Coefficients*. **SIAM Journal on Scientific Computing**, 35(6), B1215–B1234, 2013；出版社记录在线发表日期 2013-11-07。

[出版社页面与 DOI](https://epubs.siam.org/doi/10.1137/120895366)

文章研究对特征相近的线性问题复用已有 AMG 粗空间并自适应构造两层求解器，应用包含不确定渗透率的地下流问题序列。因此，“相邻 PDE 复用粗空间”与“自适应更新旧空间”本身已有先例。本项目必须和有效的传统更新方法比较，不能只比每一步从头构造。

### 2. 相似系数下选择性重算局部校正

Fredrik Hellman, Axel Målqvist, *Numerical homogenization of elliptic PDEs with similar coefficients*. arXiv:1703.08857，v1 为 2017-03-26，v2 为 2018-06-04。本次以核实的 arXiv 版本标识，未据此推断期刊出版日期。

[作者预印本](https://arxiv.org/abs/1703.08857)

文章在 PG-LOD 框架中用可计算、具有理论依据的误差指标选择性重算局部校正，并展示时变 Darcy 场景。它表明“只更新变化区域”和“用理论指标决定更新”同样不是独立的新颖性来源。本项目的局部变化界需要交代与此类更新指标的关系。

### 3. 学习多尺度延拓算子与子空间损失

Yucheng Liu, Shubin Fu, Yingjie Zhou, Changqing Ye, Eric T. Chung, *Learning a generalized multiscale prolongation operator*. arXiv:2410.06832，v1 为 **2024-10-09**，v2 为 **2025-01-13**。

[作者预印本](https://arxiv.org/abs/2410.06832)

作者学习高对比 Darcy 问题两网格预条件器中的广义多尺度延拓算子，并使用依赖系数的子空间距离处理基变换不唯一性。因而“网络预测粗基”和“子空间不变损失”均已有直接近邻。当前核对的 arXiv 页面没有期刊出版信息；不能把预印本年份改写成未经验证的 2026 年正式出版。

### 4. 学习自适应粗基、数量预测与困难局部回退

Axel Klawonn, Martin Lanser, Janine Weber-Hamacher, *Learning Adaptive Coarse Spaces Using Transferable Neural Network Models for Linear and Nonlinear Overlapping Domain Decomposition Methods*. arXiv:2607.06261，提交于 2026-07-07。

[作者预印本](https://arxiv.org/abs/2607.06261)；[全文第 4.3.2 节](https://arxiv.org/html/2607.06261v1#S4.SS3.SSS2)

工作在 AGDSW 中结合粗基回归、数量分类和后处理，并展示跨问题迁移。第 4.3.2 节及图 14 还比较了在分类为需要更多约束的边上重新解特征问题。因此，“分类器决定数量”和“困难区域回退到特征求解”不能作为本方案独有贡献。本文的接纳判据若成立，应强调它直接检查空间的数值性质，并证明其额外成本值得付出。

### 5. 从平滑误差学习粗空间

Chenglong Bao, Chen Cui, Ruohua Li, *Learning Coarse Spaces for Two-Grid Methods from Smoothed Errors with Applications to High-Contrast Porous Media Flow*. arXiv:2611.05837，提交于 2026-11-04。

[作者预印本](https://arxiv.org/abs/2611.05837)

工作用 Schwarz 平滑误差快照及能量内积 POD 建立参考空间，以连续坐标神经生成器和基不变损失学习粗空间；测试包含移动界面 Darcy 与不同网格。它要求本项目更具体地区分“相对旧空间的增量修复”，也说明移动界面、跨网格或平滑误差标签单独都不足以建立新颖性。

## 可检验的候选差异

本项目拟研究的对象是序列中的**已有空间条件下的增量方向生成**，并把独立接纳、传统修复、空间压缩与粗矩阵更新纳入同一个总成本评价。能否形成贡献取决于以下证据，而非增加模块数量：

1. 在共享检查和回退机制下，网络相对于传统候选方向确实降低在线总时间。
2. 网络学习缺失方向的收益优于直接预测完整粗空间，且旧空间输入或增量处理有必要。
3. 接纳与续用分析具有明确假设，数值实现遵守这些假设；浮点检查不被夸大为严格认证。
4. 在独立序列、拓扑变化、对比度外推以及不同右端上，收益和失败情况均被完整报告。

## 正式写作前仍需完成

阅读全文并追踪其参考文献和后续工作；查找可复现代码；核对最新版本与期刊信息；补足自适应 Schwarz/GenEO、粗空间回收和学习预条件器方向的文献。正式实验至少应复现一种最接近的学习方法与一种有效的传统自适应方法。

目前没有实验结果证明上述候选差异成立，也没有发表或分区保证。本仓库的原始实验输出和最终报告应单独保存，不能把本文件的研究假设当作已验证结论。
