
# 附录B. 参考文献和扩展阅读

- [第一章](#第一章)
- [第二掌](#第二掌)
- [第三章](#第三章)
- [第四章](#第四章)
- [第五章](#第五章)
- [第六章](#第六章)
- [第七章](#第七章)

-----

## 第一章

**正如彭博社的一个团队通过从零开始在金融数据上预训练的一个 GPT 版本所展示的那样，定制构建的 LLM 能够胜过通用 LLM。这个定制的 LLM 在金融任务上优于 ChatGPT，同时在通用 LLM 基准测试中保持了良好的性能：**

+ BloombergGPT：金融领域的大型语言模型 (2023)，吴等人著，https://arxiv.org/abs/2303.17564

<br />

**现有的 LLM 也可以通过适配和微调来胜过通用 LLM，正如 Google Research 和 Google DeepMind 的团队在医疗领域所展示的那样：**

+ 使用大型语言模型实现专家级医疗问答 (2023)，Singhal 等人著，https://arxiv.org/abs/2305.09617

<br />

**提出原始 Transformer 架构的论文：**

+ Attention Is All You Need (2017)，Vaswani 等人著，https://arxiv.org/abs/1706.03762

<br />

**最初的编码器式 Transformer，称为 BERT：**

+ BERT：用于语言理解的深度双向 Transformer 的预训练 (2018)，Devlin 等人著，https://arxiv.org/abs/1810.04805

<br />

**描述解码器式 GPT-3 模型的论文，该模型启发了现代 LLM，并将作为本书中从零开始实现 LLM 的模板：**

+ Language Models are Few-Shot Learners (2020)，Brown 等人著，https://arxiv.org/abs/2005.14165

<br />

**用于图像分类的原始 Vision Transformer，它表明 Transformer 架构不仅限于文本输入：**

+ An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020)，Dosovitskiy 等人著，https://arxiv.org/abs/2010.11929

<br />

**两种实验性的（但不太流行的）LLM 架构，它们作为并非所有 LLM 都必须基于 Transformer 架构的示例：**

+ RWKV：Transformer 时代 RNN 的革新 (2023)，Peng 等人著，https://arxiv.org/abs/2305.13048
+ Hyena Hierarchy：迈向更大的卷积语言模型 (2023)，Poli 等人著，https://arxiv.org/abs/2302.10866 Mamba：具有选择性状态空间的线性时间序列建模 (2023)，Gu 和 Dao 著，https://arxiv.org/abs/2312.00752

<br />

**Meta AI 的模型是类似 GPT 的流行实现，与 GPT-3 和 ChatGPT 相比，它是公开可用的：**

+ Llama 2：开放基础模型和微调的聊天模型 (2023)，Touvron 等人著，https://arxiv.org/abs/2307.092881

<br />

**对于对第 1.5 节中数据集参考文献感兴趣的读者，这篇论文介绍了 Eleuther AI 策划的公开可用的 The Pile 数据集：**

+ The Pile：用于语言建模的 800GB 多样化文本数据集 (2020)，Gao 等人著，https://arxiv.org/abs/2101.00027

<br />

**以下论文提供了在第 1.6 节中提及并在第 7 章中更详细讨论的用于微调 GPT-3 的 InstructGPT 的参考文献：**

+ 使用人类反馈训练语言模型以遵循指令 (2022)，Ouyang 等人著，https://arxiv.org/abs/2203.02155

<br />

## 第二掌

+ 机器学习问答 (2023)，Sebastian Raschka 著，https://leanpub.com/machine-learning-q-and-ai

<br />

**以下论文更深入地讨论了字节对编码是如何作为一种分词方法使用的：**

+ 使用子词单元进行罕见词的神经机器翻译 (2015)，Sennrich 等人著，https://arxiv.org/abs/1508.07909

<br />

**用于训练 GPT-2 的字节对编码分词器的代码已由 OpenAI 开源：**

+ https://github.com/openai/gpt-2/blob/master/src/encoder.py

<br />

**OpenAI 提供了一个交互式 Web UI 来演示 GPT 模型中的字节对分词器是如何工作的：**

+ https://platform.openai.com/tokenizer

<br />

**对于那些有兴趣从头开始编写和训练 BPE 分词器的读者，Andrej Karpathy 的 GitHub 仓库 minbpe 提供了一个最小且易于理解的实现：**

+ 一个 BPE 分词器的最小实现，https://github.com/karpathy/minbpe

<br />

**对于那些有兴趣研究其他一些流行的 LLM 使用的替代分词方案的读者，可以在 SentencePiece 和 WordPiece 的论文中找到更多信息：**

+ SentencePiece：一种用于神经文本处理的简单且与语言无关的子词分词器和反分词器 (2018)，Kudo 和 Richardson 著，https://aclanthology.org/D18-2012/
+ 快速 WordPiece 分词 (2020)，Song 等人著，https://arxiv.org/abs/2012.15524

<br />

## 第三章

**对于有兴趣了解更多关于 RNN 和语言翻译的 Bahdanau 注意力的读者，可以在以下论文中找到详细的见解：**

+ 通过联合学习对齐和翻译进行神经机器翻译 (2014)，Bahdanau、Cho 和 Bengio 著，https://arxiv.org/abs/1409.0473

<br />

**自注意力作为缩放点积注意力的概念是在最初的 Transformer 论文中提出的：**

+ Attention Is All You Need (2017)，Vaswani 等人著，https://arxiv.org/abs/1706.03762

<br />

**FlashAttention 是一种高效的自注意力机制实现，它通过优化内存访问模式来加速计算过程。FlashAttention 在数学上与标准的自注意力机制相同，但优化了计算过程以提高效率：**

+ FlashAttention：具有 IO 感知的快速且内存高效的精确注意力 (2022)，Dao 等人著，https://arxiv.org/abs/2205.14135
+ FlashAttention-2：具有更好并行性和工作分区的更快注意力 (2023)，Dao 著，https://arxiv.org/abs/2307.08691

<br />

**PyTorch 实现了一个用于自注意力和因果注意力的函数，该函数为了提高效率而支持 FlashAttention。此功能目前为测试版，可能会发生更改：**

+ `scaled_dot_product_attention` 文档：https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

<br />

**PyTorch 还实现了一个基于 `scaled_dot_product` 函数的高效 `MultiHeadAttention` 类：**

+ `MultiHeadAttention` 文档：https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

<br />

**Dropout 是一种在神经网络中使用的正则化技术，通过在训练期间随机丢弃神经网络中的单元（及其连接）来防止过拟合：**

+ Dropout：一种防止神经网络过拟合的简单方法 (2014)，Srivastava 等人著，https://jmlr.org/papers/v15/srivastava14a.html

<br />

**虽然在实践中，基于缩放点积注意力的多头注意力仍然是最常见的自注意力变体，但作者发现，即使没有值权重矩阵和投影层，也可能获得良好的性能：**

+	简化 Transformer 模块 (2023)，He 和 Hofmann 著，https://arxiv.org/abs/2311.01906

<br />

## 第四章

**这篇名为《层归一化》的论文介绍了一种技术，通过归一化隐藏层内神经元的输入总和来稳定神经网络的隐藏状态动态，与先前发表的方法相比，显著减少了训练时间：**

+ 层归一化 (2016)，Ba、Kiros 和 Hinton 著，https://arxiv.org/abs/1607.06450

<br />

**原始 Transformer 模型中使用的 Post-LayerNorm 在自注意力和前馈网络之后应用层归一化。相比之下，像 GPT-2 和更新的 LLM 中采用的 Pre-LayerNorm 在这些组件之前应用层归一化，这可以带来更稳定的训练动态，并且在某些情况下已被证明可以提高性能，如下列论文所述：**

+ 关于 Transformer 架构中的层归一化 (2020)，Xiong 等人著，https://arxiv.org/abs/2002.04745
+ ResiDual：具有双重残差连接的 Transformer (2023)，Tie 等人著，https://arxiv.org/abs/2304.14802

<br />

**由于其更高的计算效率，RMSNorm 是现代 LLM 中使用的一种流行的 LayerNorm 变体。此变体通过仅使用输入的均方根对输入进行归一化来简化归一化过程，而无需在平方之前减去均值。这意味着它在计算尺度之前不会对数据进行中心化。以下论文更详细地介绍了 RMSNorm：**

+	Root Mean Square Layer Normalization (2019)，Zhang 和 Sennrich 著，https://arxiv.org/abs/1910.07467

<br />

**GELU（高斯误差线性单元）激活函数结合了经典 ReLU 激活函数和正态分布累积分布函数的特性来建模层输出，从而在深度学习模型中实现随机正则化和非线性，如下列论文所述：**

+ 高斯误差线性单元 (GELUs) (2016)，Hendricks 和 Gimpel 著，https://arxiv.org/abs/1606.08415

<br />

**GPT-2 的论文介绍了一系列不同规模的基于 Transformer 的 LLM——参数量分别为 1.24 亿、3.55 亿、7.74 亿和 15 亿：**

+ 语言模型是无监督的多任务学习者 (2019)，Radford 等人著，https://d4mucfpksywv.cloudfront.net/better-languagemodels/language_models_are_unsupervised_multitask_learners.pdf

<br />

**OpenAI 的 GPT-3 从根本上使用了与 GPT-2 相同的架构，只不过其最大的版本（1750 亿参数）比最大的 GPT-2 模型大了 100 倍，并且在更多的数据上进行了训练。感兴趣的读者可以参考 OpenAI 的官方 GPT-3 论文以及 Lambda Labs 的技术概述，后者计算得出，在单个 RTX 8000 消费级 GPU 上训练 GPT-3 需要 665 年：**

+ 语言模型是少样本学习者 (2023)，Brown 等人著，https://arxiv.org/abs/2005.14165
+ OpenAI 的 GPT-3 语言模型：技术概述，https://lambdalabs.com/blog/demystifying-gpt-3

<br />

**NanoGPT 是一个代码仓库，其中包含一个极简但高效的 GPT-2 模型实现，类似于本书中实现的模型。虽然本书中的代码与 nanoGPT 不同，但该仓库启发了将大型 GPT Python 父类实现重组为更小的子模块：**

+	NanoGPT，一个用于训练中等规模 GPT 的仓库，https://github.com/karpathy/nanoGPT

<br />

**一篇信息丰富的博客文章指出，当上下文大小小于 32,000 个 token 时，LLM 中的大部分计算都花费在前馈层而不是注意力层：**

+ 《从长远来看（上下文）》，作者 Harm de Vries，https://www.harmdevries.com/post/context-length/

<br />

## 第五章

**作者的一个视频讲座，详细介绍了损失函数并应用对数变换以使其更易于进行数学优化：**

+ L8.2 逻辑回归损失函数，https://www.youtube.com/watch?v=GxJe0DZvydM

<br />

**以下两篇论文详细介绍了用于预训练 LLM 的数据集、超参数和架构细节：**

+ Pythia：用于分析跨训练和扩展的大型语言模型的套件 (2023)，Biderman 等人著，https://arxiv.org/abs/2304.01373
+ OLMo：加速语言模型科学 (2024)，Groeneveld 等人著，https://arxiv.org/abs/2402.00838

<br />

本书提供的以下补充代码包含从古腾堡计划准备 60,000 本公共领域书籍以用于 LLM 训练的说明：

+ 在古腾堡数据集上预训练 GPT，https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg

<br />

**第五章讨论了 LLM 的预训练，附录 D 涵盖了更高级的训练函数，例如线性预热和余弦退火。以下论文发现，类似的技术可以成功地应用于继续预训练已经预训练过的 LLM，并提供额外的技巧和见解：**

+ 简单且可扩展的持续预训练大型语言模型策略 (2024)，Ibrahim 等人著，https://arxiv.org/abs/2403.08763

<br />

**BloombergGPT 是一个领域特定的大型语言模型 (LLM) 的示例，它通过在通用和领域特定的文本语料库（特别是金融领域）上进行训练而创建：**

+ BloombergGPT：金融领域的大型语言模型 (2023)，吴等人著，https://arxiv.org/abs/2303.17564

<br />

**GaLore 是一个旨在提高 LLM 预训练效率的最新研究项目。所需的代码更改非常简单，只需将训练函数中 PyTorch 的 AdamW 优化器替换为 galore-torch Python 包提供的 GaLoreAdamW 优化器即可。**

+ GaLore：通过梯度低秩投影实现内存高效的 LLM 训练 (2024)，Zhao 等人著，https://arxiv.org/abs/2403.03507
+ GaLore 代码仓库，https://github.com/jiaweizzhao/GaLore

<br />

**以下论文和资源分享了公开可用的大规模 LLM 预训练数据集，这些数据集包含数百 GB 到数 TB 的文本数据：**

+ Dolma：一个用于 LLM 预训练研究的 3 万亿 token 的开放语料库，Soldaini 等人，2024 年，https://arxiv.org/abs/2402.00159
+ The Pile：一个用于语言建模的 800GB 多样化文本数据集，Gao 等人，2020 年，https://arxiv.org/abs/2101.00027
+ The RefinedWeb Dataset for Falcon LLM：仅使用网络数据超越精心策划的语料库，Penedo 等人 (2023)，https://arxiv.org/abs/2306.01116
+ RedPajama，Together AI，https://github.com/togethercomputer/RedPajama-Data
+ The FineWeb dataset，包含超过 15 万亿 token 的来自 CommonCrawl 的清洗和去重后的英语网络数据，https://huggingface.co/datasets/HuggingFaceFW/fineweb

<br />

**最初介绍 top-k 采样的论文：**

+ Hierarchical Neural Story Generation，Fan 等人 (2018)，https://arxiv.org/abs/1805.04833

<br />

**集束搜索（第五章未涵盖）是一种替代的解码算法，它通过在每个步骤仅保留得分最高的局部序列来生成输出序列，以平衡效率和质量：**

+ Diverse Beam Search：从神经序列模型解码多样化解，Vijayakumar 等人 (2016)，https://arxiv.org/abs/1610.02424

<br />

## 第六章

**讨论不同类型微调的额外资源：**

+ 使用和微调预训练的 Transformer，https://magazine.sebastianraschka.com/p/using-and-finetuning-pretrained-transformers
+ 微调大型语言模型，https://magazine.sebastianraschka.com/p/finetuning-large-language-models

<br />

**其他实验，包括对微调第一个输出 token 与最后一个输出 token 的比较，可以在 GitHub 上的补充代码材料中找到：**

+ 额外的垃圾邮件分类实验，https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06/02_bonus_additional-experiments

<br />

**对于二元分类任务（例如垃圾邮件分类），从技术上讲，只使用一个输出节点而不是两个输出节点是可行的，正如我在以下文章中讨论的那样：**

+ 损失函数学习——优化 PyTorch 中的负对数似然和交叉熵，https://sebastianraschka.com/blog/2022/losses-learned-part1.html

<br />

**你可以在以下文章中找到关于微调 LLM 不同层的额外实验，该文章表明，除了输出层之外，微调最后一个 Transformer 模块可以显著提高预测性能：**

+ 微调大型语言模型，https://magazine.sebastianraschka.com/p/finetuning-large-language-models

<br />

**读者可以在 imbalanced-learn 的文档中找到处理不平衡分类数据集的额外资源和信息：**

+	Imbalanced-learn 用户指南，https://imbalanced-learn.org/stable/user_guide.html

<br />

**对于有兴趣对垃圾邮件电子邮件而不是垃圾短信进行分类的读者，以下资源提供了一个大型电子邮件垃圾邮件分类数据集，其格式与第 6 章中使用的数据集格式类似的便捷 CSV 格式：**

+ 电子邮件垃圾邮件分类数据集，https://huggingface.co/datasets/TrainingDataPro/email-spam-classification

<br />

**GPT-2 是一种基于 Transformer 架构解码器模块的模型，其主要目的是生成新的文本。作为替代方案，诸如 BERT 和 RoBERTa 之类的基于编码器的模型对于分类任务可能更有效：**

+ BERT：用于语言理解的深度双向 Transformer 的预训练 (2018)，Devlin 等人著，https://arxiv.org/abs/1810.04805
+ RoBERTa：一种鲁棒优化的 BERT 预训练方法 (2019)，Liu 等人著，https://arxiv.org/abs/1907.11692
+ 对 5 万条 IMDB 电影评论进行情感分类的额外实验，https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/03_bonus_imdb-classification

<br />

**最近的论文表明，通过在分类微调过程中移除因果掩码并进行其他修改，可以进一步提高分类性能：** 

+ Label Supervised LLaMA Finetuning (2023)，Li 等人著，https://arxiv.org/abs/2310.01208
+ LLM2Vec：大型语言模型是隐藏的强大文本编码器 (2024)，BehnamGhader 等人著，https://arxiv.org/abs/2404.05961

<br />

## 第七章

**用于指令微调的 Alpaca 数据集包含 5.2 万个指令-响应对，是首批最受欢迎的公开指令微调数据集之一：**

+	Stanford Alpaca：一个遵循指令的 Llama 模型，https://github.com/tatsu-lab/stanford_alpaca

<br />

**以下列出的是适合指令微调的额外公开数据集：**

+ LIMA，https://huggingface.co/datasets/GAIR/lima；包含一千个高质量的指令-响应对；更多信息请参阅论文《LIMA: Less Is More for Alignment》(2023)，https://arxiv.org/abs/2305.11206
+ UltraChat，https://huggingface.co/datasets/openchat/ultrachat-sharegpt；一个包含 80.5 万个指令-响应对的大规模数据集；更多信息请参阅论文《Enhancing Chat Language Models by Scaling High-quality Instructional Conversations》(2023)，https://arxiv.org/abs/2305.14233
+ Alpaca GPT4，https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json，一个类似 Alpaca 的数据集，包含 5.2 万个使用 GPT-4 而非 GPT-3.5 生成的指令-响应对

<br />

**Phi-3 是一个拥有 38 亿参数的模型，其指令微调变体据称可与更大的专有模型（如 GPT-3.5）相媲美：**

+ Phi-3 技术报告：一款可在您的手机本地运行的高性能语言模型 (2024)，Abdin 等人著，https://arxiv.org/abs/2404.14219

<br />

**研究人员提出了一种合成指令数据生成方法，该方法从一个指令微调的 Llama-3 模型生成 30 万个高质量的指令-响应对。在一个预训练的 Llama 3 基础模型上，使用这些指令示例进行微调后，其性能与原始的指令微调 Llama-3 模型相当：**

+	Magpie：通过提示对齐的 LLM 从零开始合成对齐数据 (2024)，Xu 等人著，https://arxiv.org/abs/2406.08464

<br />

**研究表明，在指令微调中不屏蔽指令和输入可以有效地提高在各种 NLP 任务和开放式生成基准上的性能，尤其是在使用包含长指令和简短输出的数据集或使用少量训练示例进行训练时：**

+ Instruction Tuning With Loss Over Instructions (2024)，Shi 著，https://arxiv.org/abs/2405.14394

<br />

**Prometheus 和 PHUDGE 是公开可用的大型语言模型，它们在评估具有可自定义标准的长篇回复方面与 GPT-4 相媲美。我们在第 7 章中没有使用这些模型，因为 Ollama 尚不支持它们，因此无法在笔记本电脑上高效执行。**

+ Prometheus：在语言模型中引入细粒度的评估能力 (2023)，Kim 等人著，https://arxiv.org/abs/2310.08491
+ PHUDGE：将 Phi-3 作为可扩展的评判者 (2024)，Deshwal 和 Chawla 著，https://arxiv.org/abs/2405.08029
+ Prometheus 2：一个专门评估其他语言模型的开源语言模型 (2024)，https://arxiv.org/abs/2405.01535

<br />

**以下报告中的结果支持这样一种观点：大型语言模型主要在预训练期间获取事实知识，而微调主要提高它们使用这些知识的效率。此外，这项研究探讨了使用新的事实信息对大型语言模型进行微调如何影响它们使用现有知识的能力，揭示了模型学习新事实的速度较慢，并且在微调期间引入新事实会增加模型生成不正确信息的倾向：**

+ 在新的知识上微调 LLM 是否会鼓励幻觉？(2024)，Gekhman 著，https://arxiv.org/abs/2405.05904

<br />

**偏好微调是指令微调之后的一个可选步骤，旨在使 LLM 更紧密地与人类偏好对齐。作者的以下文章提供了有关此过程的更多信息：**

+ LLM 训练：RLHF 及其替代方案，https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives
+ LLM 预训练和奖励模型评估技巧，https://sebastianraschka.com/blog/2024/research-papers-in-march2024.html







