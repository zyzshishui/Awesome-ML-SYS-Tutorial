# Kimi K1.5: Long Context RL 的成功实践

感谢 kimi 团队的这篇雄文。大概在 DeepSeek 开源 R1 的同一时间，就有许多朋友向我推荐过 K1.5 的技术报告。苦于工作繁琐，一直没有空拜读。正好最近去西雅图出差，在 LAX 往返 SEA 的路上终于有时间虔诚地读完这一工作。越读起来，越有一种文人相赏，难逢知音的感觉。

本科毕业以来，确实有感受到自己在团队协作和个人能力上的长足长进。但是博士入学之后，我一直没有什么高强度投入的工作发表，未免感到焦虑。今天读完这样的雄文，心情大爽。希望自己能在余下的科研生涯中多参与这样具有开源精神的重磅工作。能让自己的名字出现在此番工作的作者名录之上，不比多发几篇 XXX 或者 XXXX 的论文强？当然，这就又带来了在大项目中，如何证明自己 credit 的问题。不过，我总归相信自己的想法仍是大有裨益的。

絮絮叨叨说了这么多，这篇文章主要复盘自己拜读 K1.5 技术报告的思索。由于是技术报告，这篇扎实的文章涵盖了从数据、训练方法到训练系统的方方面面，读完真是余音绕梁，不绝如缕。

## RL Recipe

K1.5 的训练可以细分为 pretrain，vanila SFT，long-CoT SFT 和 RL 四个阶段。技术报告主要讲述的是 RL 阶段的故事。

### RL prompt 选择

高质量的 RL prompt 需要 diver, balance and accurate to evaluate。为了决定每个 prompt 的难度，作者采用一个 SFT 模型在较高的 temperature 下生成 10 次答案，以 10 次内的通过率作为难度。此外，一些复杂的推理问题可能通过错误的推理过程也能得出正确答案。为了避免此类 reward hacking，作者进一步确保每个 prompt 的 reasoning path 和 final answer 都能被准确验证。作者先排除了容易出现此类错误的题目，例如多选题、判断题和证明题。然后，作者进一步过滤掉一些容易猜测出答案的问题。具体来说，给模型 8 次机会，如果在没有 CoT 的情况下，有超过 1 次可以直接给出答案，就将其移除。

### Long-CoT SFT

作者通过 prompt engineering 构建了一个 multi-modal long-CoT warmup dataset，来让模型初步学会这几种推理能力，evaluation，reflection，exploration 和 planning。

### Length Penalty

在进行 long context RL 训练的过程中，如果不对模型的输出长度做出控制，会很容易观测到 answer length 的显著增加。虽然这带来了更好的性能，但过长的推理过程在训练和推理时成本高昂，而且人类通常不倾向于过度思考。因此，作者引入了一个长度惩罚项，来控制模型的输出长度：

$$
\mathrm{len\_reward}(i) = \begin{cases} \lambda & \text{if } r(x, \mathrm{y_i}, y^*) = 1 \\ \min(0, \lambda) & \text{if } r(x, \mathrm{y_i}, y^*) = 0 \end{cases}
$$

$$
\lambda = 0.5 - \frac{\text{len}(i) - \mathrm{min\_len}}{\mathrm{max\_len} - \mathrm{min\_len}}
$$

分开想想这几种情况：

1. 最长的推理过程，正确的答案。$len\_reward = -0.5$，抑制模型生成此推理过程。
2. 最长的推理过程，错误的答案。$len\_reward = -0.5$，抑制模型生成此推理过程。
3. 最短的推理过程，正确的答案。$len\_reward = 0.5$，鼓励模型生成此推理过程。
4. 最短的推理过程，错误的答案。$len\_reward = 0$，对模型没有影响。

对于长度在 $\frac{\mathrm{max\_len} + \mathrm{min\_len}}{2}$ 以下的的推理过程，如果答案错误，则 length reward 为 0，如果答案正确，则 length reward 大于 0 且随着 length 递减。超过这个长度的 reasoning path，无论答案正误与否，都给一样的负分 length reward。

### 采样策略

虽然 RL 本身具有较好的采样特性，也即难度更高的问题会提供更大的梯度，但其训练效率仍然有限。一些明确的先验采样方法可能会带来更大的性能提升。作者采用的方案有：

1. 课程学习：从较简单的任务开始训练，逐步过渡到更具挑战性的任务。由于初始 RL 模型的性能有限，在非常困难的问题上花费有限的计算预算往往只能产生很少的正确样本，从而导致较低的训练效率。同时，用于训练的数据天然就包含难度标签，进一步让难度递增学习策略直观且有效。
2. 优先采样：成功率较低的问题采样更多次。我们跟踪每个问题 $i$ 的成功率 $s_i$，并按照 $1 - s_i$ 的比例采样问题，从而使成功率较低的问题获得更高的采样概率。这将模型的努力集中在它最薄弱的领域，从而加速学习并提升整体性能。

### Code, Math and Visual Data

由于爬虫的合规性限制，很多网上得到的 coding 问题没有测例。为此，作者使用 CYaRon1 来生成测例，作为 reward。这当然，需要很多严苛的假设，譬如不需要特殊的评判，并且这些问题有可用的 ground truth solution，以便利用这些 solution 生成更高质量的测试用例。

至于数学，评估数学问题的一大挑战是，不同的表达形式可能代表相同的答案。例如，$a^2 - 4$ 和 $(a + 2)(a - 2)$ 可能都是同一个问题的有效解。为了提高奖励模型的评分准确性，作者尝试了两种方法：

1. Classic RM：参考 InstructGPT，作者实现了一个基于 value-head 的奖励模型，并收集了约 80 万个数据点进行微调。该模型最终以 question, answer, reference answer 作为输入，输出一个标量，指示响应是否正确。
2. Chain-of-Thought RM：最近的一些工作证明，加入了 CoT 的 reward model 在需要细微正确性标准的任务上显著更优。作者继续收集了约 80 万个带有 CoT 的数据集来微调 reward model。在与 Classic RM 相同输入的基础上，CoT model 明确生成逐步推理过程，最后以 JSON 格式提供最终的 reward。在作者的手动抽查中，Classic RM 的准确率约为 84.4，而 CoT RM 达到了 98.5 的准确率。最终，他们采用了 CoT RM 以确保更正确的反馈。

Vision RL 的数据主要来源于三大类：Real-world data，Synthetic visual reasoning data 以及 Text-rendered data。其中，text-rendered data 的做法颇有意思，将文本内容转换为图片，专门强调了模型处理文本密集图像的能力。


### Long2Short 训练

Long-CoT 模型的推理开销显著更大，作者指出了多种可以将 Long CoT 能力迁移到 Short CoT 上。比如 model merging，shortest rejection sampling，DPO 还有 Long2Short RL。这里首先分享一个最有趣的方法——model merging。非常简单，直接把 long cot 模型和 short cot 模型的参数取平均值，就得到了更好的 short cot 模型。再者，shortest rejection sampling 就是每次从所有正确的 samples 中，选择最短且正确的答案作为最终的 sample。当然，rejection sampling 的另一面就是 DPO，可以把短的错误答案和正确的长答案都作为 negative sample，构造 pairewise preference data。同样，简单的 rejection sampling 也可以广泛用于 math 和 coding 问题上，因为 rule-based verification 往外比起人自身还准确。在 SFT 阶段，作者也利用 rejection sampling 来扩充数据集。

## RL Infra

这是我最感兴趣的部分了，作者在他们的 RL 系统中重点讲述了 partial rollout 这一想法，这是最重要的创新了。

### Partial Rollout

考虑真实场景下的大规模多任务 RL。这些 RL 任务在 rollout 阶段的 decode 长度差距非常大，譬如 reasoning task 可能天然就比选择题需要 decode 的 tokens 多的多的多，然后即便是同一类问题，长文本翻译和短文本翻译需要的 decode 长度也不太一样。为此，进一步考虑一个 PPO iteration，如果我们要获得 128 个 samples 用于这一步的参数更新，我们可以直接从 prompt pool 中采样 128 个 input，然后发送给 rollout engine，让其完整地从 prefill 结束后的第一个 token 开始 decode，直到 128 个 input 都完成 decode。这些 input 可能来自不同任务，其需要的 decode 长度差距很大。假设 rollout engine 具有简单的 routing 分发机制，可能会按照 prefix maxium 的方式，将 128 个 input 分到多个 worker 上，每个 worker 各自完成各自的 decode 任务，最后将结果返回给 PPO 训练。如果一个 request 需要 decode 的内容特别长，那么我们可以想象，其相似 prefix 的 request 可能需要 decode 的内容也特别长。很不巧，按照我们的 routing 策略，这两个 request 很有可能被分到同一个 worker 上。虽然这样的 routing 策略节省了 prefill 开销，但是大量的 long decode request 都发到了同一个 worker 上，这个 worker 事实上会成为所有 rollout workers 中的 bottleneck。

为了解决这个问题，作者提出了他们的 partial rollout 策略，而我在他们的基础上分享一些我听到的可行做法。还是考虑我要 128 个 samples 的场景，首先为这个 128 个 samples 的 expected length 在 iteration t 设定一个阈值 $\text{Sample Range} = L_{t, min}, L_{t, max}$，然后进行超采样，为了得到 128 个 examples，我们实际上输入给 rollout engine 的请求数量为 512 个。超采样的坏处显而易见，毕竟多采样了这么多 examples，速度肯定是变慢的。不过，考虑到 rollout engine 的 batch size 可以非常大，实际上很多 dp worker 分下来，batch size 调大，单个 request 的处理速度不会收到特别大的影响。而且，考虑了 sample range 后，这 512 个 examples 并发同时开始 decode，我们把先在 sample range 内成功 decode 的 request 返回给 PPO 训练，剩下的要么继续 decode（因为 k1.5 的 RL 系统是 rollout 和 training speerate 的，所以在 training 的时候不用回收 rollout engine，所以可以继续 decode），要么就 cache 存下来，这个 iteration 不继续 decode 了。有了这种设计，发挥想象力，在下个 iteration，首先是 policy model 的参数和 sample range 更新了，再者是前一个 iteration 没有在 sample range 内完成 decode 的 request 可以继续 decode，如果成功了，就返回给 PPO 训练，否则还是 cache 存下来。

这个策略确保了每一个 iteration 返回的 samples 的 decode 长度都在 Sample Range 内，对于训练而言是更加稳定的。也能一定程度缓和 rollout engine 的 load balance 问题。在业界的实践看来，是非常实在的方法。然而，我其实想到很多悬而未决的问题：

1. 显然，partial rollout 会破坏 on-policy 属性。很有可能一个 request 再被用于训练时，decode 出来的部分会分 chunk 来自不同的 iteration。比如 0～64k 来自 iteration t-2 ，64k～96k 来自 iteration t-1，而从 96k 开始到结束，才真正来自 iteration t。这在理论上能够证明有相近的效果么？
2. 没有 decode 结束的 request 会被 cache 下来，但是下一轮显然会加入新的 request。前一轮的 request 已经有相当一部分被 cache 下来，这些新的 request 需要 decode 的量为了达到同样的 sample range，显然是更多的，这样是不是就造就了新的 load unbalance 呢？

### RL framework

作者搭建了基于 megatron 和 vllm 的大规模 RL 系统。在这种工业级实践中，他们关注的和我们确实区别很大。比如，他们会关注 rollout stage 和 training stage 之间的启动间隔，从训练到 rollout 需要 1min，反过来需要 10s。此外，还涉及到了我从没考虑过的变量——checkpoint engine。简单来说，随着 rollout 的逐步进行，需要的 trajactoris length 越长，rollout engine 开始设置的 context length 可能就不够大了。目前的做法是反复 kill and relaunch 新的 rollout engine，这需要存下 ckpt 并且尽可能降低重启的开销。

在 code 执行方面，他们采用了 crun 而不是 dokcer，效率直观上快了很多。

## Ablation

- 大模型短推理和小模型长推理：小模型可以用长推理来比肩大模型，但是总体上，大模型的 token efficiency 是更好的。large model with short context 和 small model with long context 目前是效果一致的方案。
- 与 ReST 的方法相反，在训练中引入 negative gradient 能够显著增强模型生成 long cot 的效率。
- 课程学习（由易到难学习）带来了显著的提升。