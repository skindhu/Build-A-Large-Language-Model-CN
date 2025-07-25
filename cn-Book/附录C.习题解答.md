
# 附录C. 习题解答

- [第二掌](#第二掌)
  - [练习 2.1](#练习-21)
  - [练习 2.2](#练习-22)
- [第三章](#第三章)
  - [练习 3.1](#练习-31)
  - [练习 3.2](#练习-32)
  - [练习 3.3](#练习-33)
- [第四章](#第四章)
  - [练习 4.1](#练习-41)
  - [练习 4.2](#练习-42)
- [第五章](#第五章)
  - [练习 5.1](#练习-51)
  - [练习 5.2](#练习-52)
  - [练习 5.3](#练习-53)
  - [练习 5.4](#练习-54)
  - [练习 5.5](#练习-55)
  - [练习 5.6](#练习-56)
- [第六章](#第六章)
  - [练习 6.1](#练习-61)
  - [练习 6.2](#练习-62)
  - [练习 6.3](#练习-63)
- [第七章](#第七章)
  - [练习 7.1](#练习-71)
  - [练习 7.2](#练习-72)
  - [练习 7.3](#练习-73)
  - [练习 7.4](#练习-74)

-----
<br />
练习答案的完整代码示例可以在补充 GitHub 仓库中找到：https://github.com/rasbt/LLMs-from-scratch。

<br />

## 第二掌

### 练习 2.1

你可以通过一次用一个字符串提示编码器来获取单独的 token ID：

```python
print(tokenizer.encode("Ak"))
print(tokenizer.encode("w"))
# ...
```

打印如下：

```python
[33901]
[86]
# ...
```

然后你可以使用以下代码来组装原始字符串：

```python
print(tokenizer.decode([33901, 86, 343, 86, 220, 959]))
```

打印如下：

```python
'Akwirw ier'
```

<br />

### 练习 2.2

具有 max_length=2 和 stride=2 的数据加载器的代码：

```python
dataloader = create_dataloader(raw_text, batch_size=4, max_length=2, stride=2)
```

它产生以下格式的批次：

```python
tensor([[ 40, 367],
        [2885, 1464],
        [1807, 3619],
        [ 402, 271]])
```

第二个数据加载器的代码，其 max_length=8，stride=2：

```python
dataloader = create_dataloader(raw_text, batch_size=4, max_length=8, stride=2)
```

一个示例批次如下所示：

```python
tensor([[ 40, 367, 2885, 1464, 1807, 3619, 402, 271],
        [ 2885, 1464, 1807, 3619, 402, 271, 10899, 2138],
        [ 1807, 3619, 402, 271, 10899, 2138, 257, 7026],
        [ 402, 271, 10899, 2138, 257, 7026, 15632, 438]])
```

<br />

## 第三章

### 练习 3.1

正确的权重分配如下：

```python
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
```

<br />

### 练习 3.2

为了获得与单头注意力中相同的 2 维输出维度，我们需要将投影维度 d_ou t更改为 1 。

```python
d_out = 1
mha = MultiHeadAttentionWrapper(d_in, d_out, block_size, 0.0, num_heads=2)
```

<br />

### 练习 3.3

最小 GPT-2 模型的初始化如下：

```python
block_size = 1024
d_in, d_out = 768, 768
num_heads = 12
mha = MultiHeadAttention(d_in, d_out, block_size, 0.0, num_heads)
```

<br />

## 第四章

### 练习 4.1

我们可以按如下方式计算前馈模块和注意力模块中的参数数量：

```python
block = TransformerBlock(GPT_CONFIG_124M)
total_params = sum(p.numel() for p in block.ff.parameters())
print(f"Total number of parameters in feed forward module: {total_params:,}")
total_params = sum(p.numel() for p in block.att.parameters())
print(f"Total number of parameters in attention module: {total_params:,}")
```

正如我们所见，前馈模块包含的参数数量大约是注意力模块的两倍：

```python
Total number of parameters in feed forward module: 4,722,432
Total number of parameters in attention module: 2,360,064
```

<br />

### 练习 4.2

要实例化其他GPT模型尺寸，我们可以修改配置字典为如下所示（此处以GPT-2 XL为例）：

```python
GPT_CONFIG = GPT_CONFIG_124M.copy()
GPT_CONFIG["emb_dim"] = 1600
GPT_CONFIG["n_layers"] = 48
GPT_CONFIG["n_heads"] = 25
model = GPTModel(GPT_CONFIG)
```

然后，重用第 4.6 节中的代码来计算参数数量和 RAM 需求，我们得到以下结果：

```python
gpt2-xl:
Total number of parameters: 1,637,792,000
Number of trainable parameters considering weight tying: 1,557,380,800
Total size of the model: 6247.68 MB
```

<br />

## 第五章

### 练习 5.1

我们可以使用本节中定义的 `print_sampled_tokens` 函数来打印 token（或单词）“pizza” 被采样的次数。让我们从我们在 5.3.1 节中定义的代码开始。

如果温度为 0 或 0.1，则 “pizza” token 被采样 0 次；如果温度升高到 5，则被采样 32 次。估计的概率是 32/1000 × 100% = 3.2%。实际概率是 4.3%，包含在重新缩放的 softmax 概率张量中 (scaled_probas[2][6])。

<br />

### 练习 5.2

Top-k 采样和温度缩放是需要根据 LLM 以及输出中所需的 diversity 和随机性程度进行调整的设置。

当使用相对较小的 top-k 值（例如，小于 10）并且温度设置为低于 1 时，模型的输出变得不那么随机，更具确定性。当我们希望生成的文本更具可预测性、连贯性，并且更接近基于训练数据的最可能结果时，这种设置非常有用。

这种低 k 值和温度设置的应用包括生成正式文档或报告，在这些场景中，清晰度和准确性最为重要。其他应用示例包括技术分析或代码生成任务，在这些任务中，精确性至关重要。此外，问答和教育内容需要准确的答案，低于 1 的温度有助于实现这一点。

另一方面，较大的 top-k 值（例如，范围在 20 到 40 之间）和高于 1 的温度值很有用，当使用 LLM 进行头脑风暴或生成创意内容（如小说）时。

<br />

### 练习 5.3

有多种方法可以使用 `generate` 函数强制确定性行为：

1. 将 top_k 设置为 None 且不应用温度缩放；
2. 将 top_k 设置为 1。

<br />

### 练习 5.4

本质上，我们必须加载我们在主章节中保存的模型和优化器：

```python
checkpoint = torch.load("model_and_optimizer.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

然后，调用 `train_simple_function` 并设置 `num_epochs=1`，以再次训练模型一个 epoch。

<br />

### 练习 5.5

我们可以使用以下代码来计算 GPT 模型的训练集和验证集损失：

```python
train_loss = calc_loss_loader(train_loader, gpt, device)
val_loss = calc_loss_loader(val_loader, gpt, device)
```

具有 1.24 亿参数的模型得到的损失如下：

```python
Training loss: 3.754748503367106
Validation loss: 3.559617757797241
```

主要的观察结果是，训练集和验证集的性能处于同一水平。这可能有多种解释。

1. 当 OpenAI 训练 GPT-2 时，“The Verdict” 并非预训练数据集的一部分。因此，该模型并没有显式地过拟合训练集，并且在 “The Verdict” 的训练集和验证集部分上表现得同样出色。（验证集损失略低于训练集损失，这在深度学习中是不常见的。然而，这很可能是由于数据集相对较小而产生的随机噪声。在实践中，如果没有过拟合，训练集和验证集的性能预计大致相同。）
2. “The Verdict” 是 GPT-2 训练数据集的一部分。在这种情况下，我们无法判断模型是否过拟合训练数据，因为验证集也可能被用于训练。为了评估过拟合的程度，我们需要一个在 OpenAI 完成 GPT-2 的训练后生成的新数据集，以确保它不可能是预训练数据的一部分。

<br />

### 练习 5.6

在主章节中，我们使用了最小的 GPT-2 模型，它只有 1.24 亿个参数。原因是尽可能降低资源需求。然而，你可以通过极少的代码更改轻松地尝试更大的模型。例如，在第 5 章中，我们加载的是 15.58 亿个参数的模型权重而不是 1.24 亿个，我们只需要更改以下两行代码：

```python
hparams, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
model_name = "gpt2-small (124M)"
```

 更新后的代码如下：

```python
hparams, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2")
model_name = "gpt2-xl (1558M)"
```

<br />

## 第六章

### 练习 6.1

我们可以通过在初始化数据集时将最大长度设置为 `max_length = 1024`，来将输入填充到模型支持的最大 token 数：

```python
train_dataset = SpamDataset(..., max_length=1024, ...)
val_dataset = SpamDataset(..., max_length=1024, ...)
test_dataset = SpamDataset(..., max_length=1024, ...)
```

然而，额外的填充导致测试准确率大幅下降，仅为 78.33%（相比之下，主章节中的准确率为 95.67%）。

<br />

### 练习 6.2

与其仅微调最后一个 Transformer 模块，我们可以通过从代码中删除以下几行来微调整个模型：

```python
for param in model.parameters():
		param.requires_grad = False
```

此修改使测试准确率提高了 1%，达到 96.67%（相比之下，主章节中的准确率为 95.67%）。

<br />

### 练习 6.3

与其微调最后一个输出 token，我们可以通过将代码中所有出现的 `model(input_batch)[:, -1, :]` 更改为 `model(input_batch)[:, 0, :]` 来微调第一个输出 token。

正如预期的那样，由于第一个 token 包含的信息比最后一个 token 少，这一更改导致测试准确率大幅下降至 75.00%（相比之下，主章节中的准确率为 95.67%）。

<br />

## 第七章

### 练习 7.1

Phi-3 的提示格式，如图 7.4 在第 7 章中所示，对于给定的输入示例，看起来如下：

```python
<user>
Identify the correct spelling of the following word: 'Occasion'
<assistant>
The correct spelling is 'Occasion'.
```

要使用此模板，我们可以按如下方式修改 `format_input` 函数：

```python
def format_input(entry):
    instruction_text = (
    f"<|user|>\n{entry['instruction']}"
    )
    input_text = f"\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
```

最后，当我们收集测试集响应时，我们还必须更新提取生成响应的方式：

```python
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    tokenizer=tokenizer
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (                                          #A
        generated_text[len(input_text):]
        .replace("<|assistant|>:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text
    
    
#A New: Adjust ###Response to <|assistant|>
```

使用 Phi-3 模板对模型进行微调大约快 17%，因为它使得模型输入更短。得分接近 50，这与我们之前使用 Alpaca 风格的提示所获得的分数大致相同。

<br />

### 练习 7.2

为了像第 7 章图 7.13 中所示那样屏蔽指令，我们需要对 `InstructionDataset` 类和 `custom_collate_fn` 函数进行一些小的修改。我们可以修改 `InstructionDataset` 类来收集指令的长度，我们将在 collate 函数中使用这些长度，以便在编写 collate 函数时定位目标中的指令内容位置，如下所示：

```python
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.instruction_lengths = []                                     #A
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
        
        self.encoded_texts.append(
       		 tokenizer.encode(full_text)
        )
        instruction_length = len(tokenizer.encode(instruction_plus_input))
        self.instruction_lengths.append(instruction_length)                #B
    
    def __getitem__(self, index):                                          #C
    		return self.instruction_lengths[index], self.encoded_texts[index]
  
    def __len__(self):
    		return len(self.data)
      
      
#A 用于指令长度的单独列表
#B 收集指令长度
#C 分别返回指令长度和文本
```

接下来，我们更新 `custom_collate_fn`，由于 `InstructionDataset` 数据集的更改，现在的每个批次都是一个包含 `(instruction_length, item)` 的元组，而不仅仅是 `item`。此外，我们现在屏蔽目标 ID 列表中的相应指令 token：

```python
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):

batch_max_length = max(len(item)+1 for instruction_length, item in batch)      #A
inputs_lst, targets_lst = [], []

for instruction_length, item in batch:                                         #A
    new_item = item.copy()
    new_item += [pad_token_id]
    padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
    inputs = torch.tensor(padded[:-1])
    targets = torch.tensor(padded[1:])
    mask = targets == pad_token_id
    indices = torch.nonzero(mask).squeeze()
    if indices.numel() > 1:
    		targets[indices[1:]] = ignore_index
        
    targets[:instruction_length-1] = -100                                       #B
    
    if allowed_max_length is not None:
        inputs = inputs[:allowed_max_length]
        targets = targets[:allowed_max_length]
    
    inputs_lst.append(inputs)
    targets_lst.append(targets)
    
inputs_tensor = torch.stack(inputs_lst).to(device)
targets_tensor = torch.stack(targets_lst).to(device)

return inputs_tensor, targets_tensor


#A 批次现在是一个元组
#B 屏蔽目标中的所有输入和指令 token
```

当评估使用这种指令屏蔽方法进行微调的模型时，它的性能略有下降（使用第 7 章中的 Ollama Llama 3 方法评估，大约下降 4 分）。这与论文《Instruction Tuning With Loss Over Instructions》（https://arxiv.org/abs/2405.14394）中的观察结果一致。

<br />

### 练习 7.3

为了在原始的 Stanford Alpaca 数据集（https://github.com/tatsulab/stanford_alpaca）上微调模型，我们只需要将文件 URL 从：

```python
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_mainchapter-code/instruction-data.json"
```

修改成：

```python
url = "https://raw.githubusercontent.com/tatsulab/stanford_alpaca/main/alpaca_data.json"
```

请注意，该数据集包含 5.2 万条记录（是第 7 章中的 50 倍），并且记录比我们在第 7 章中使用的更长。因此，强烈建议在 GPU 上运行训练。如果遇到内存不足的错误，请考虑将批处理大小从 8 减少到 4、2 或 1。除了降低批处理大小之外，您可能还需要考虑将 `allowed_max_length` 从 1024 降低到 512 或 256。

以下是 Alpaca 数据集中的一些示例，包括生成的模型回复。

<br />

### 练习 7.4

要使用 LoRA 对模型进行指令微调，请使用附录 E 中的相关类和函数：

```python
from appendix_E import LoRALayer, LinearWithLoRA, replace_linear_with_lora
```

接下来，在第 7.5 节的模型加载代码下方添加以下代码行：

```python
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters before: {total_params:,}")

for param in model.parameters():
		param.requires_grad = False
    
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after: {total_params:,}")
replace_linear_with_lora(model, rank=16, alpha=16)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")
model.to(device)
```

请注意，在 Nvidia L4 GPU 上，使用 LoRA 进行微调在 L4 上运行需要 1.30 分钟。在同一 GPU 上，原始代码运行需要 1.80 分钟。因此，在这种情况下，LoRA 大约快 28%。使用第 7 章中的 Ollama Llama 3 方法评估的分数约为 50，这与原始模型的分数大致相同。

