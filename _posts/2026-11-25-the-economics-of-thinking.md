---
layout: post
title: "The Economics of Thinking: How Reasoning Models Turn Test-Time Compute into Better Decisions"
subtitle: "A deep technical guide to reasoning LLMs, reinforcement learning, search, verifiers, compute-optimal inference, and production trade-offs"
description: "An end-to-end systems and research guide to reasoning models: test-time compute, chain-of-thought, RL, best-of-N, self-consistency, process verifiers, tree search, adaptive budgets, evaluation, latency, costs, and a runnable Python demonstration."
meta-title: "The Economics of Thinking | Reasoning Models, Search, RL, and Test-Time Compute"
meta-description: "Inside reasoning AI: how inference-time search, verification, and reinforcement learning improve difficult decisions—and when more thinking is not worth the cost."
date: 2026-11-25
published: true
use_math: true
# Load MathJax for math rendered in this older Jekyll theme.
ext-js:
  - "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
image: "/assets/img/blog/reasoning-models-test-time-compute-cover.jpg"
thumbnail-img: "/assets/img/blog/reasoning-models-test-time-compute-cover.jpg"
share-img: "https://bilwasiva.github.io/assets/img/blog/reasoning-models-test-time-compute-cover.jpg"
tags:
  - Artificial Intelligence
  - Machine Learning
  - Large Language Models
  - Reasoning Models
  - Reinforcement Learning
  - Test-Time Compute
  - AI Research
  - Systems Engineering
  - Generative AI
---

<!-- Jekyll's post layout supplies the page H1. The post starts with a standalone hero and lead. -->

*What happens when an AI system can spend more computation on a difficult question—and how should we decide whether the extra thinking is worth it?*

![The Economics of Thinking: reasoning models, verifiers, search, and test-time compute]({{ '/assets/img/blog/reasoning-models-test-time-compute-cover.jpg' | relative_url }})

> **The central idea:** More computation can improve reasoning, but only when that computation is directed toward productive exploration, grounded verification, and sensible stopping decisions. **More tokens are not synonymous with more intelligence.**

An ordinary language-model request might look simple: a user asks a question, the model generates tokens, and a final answer appears. But difficult questions rarely behave like simple autocomplete. An advanced mathematics problem can require a sequence of dependent deductions. A coding task may demand candidate algorithms, implementations, tests, and debugging. A scientific question may require checking assumptions, reconciling evidence, and rejecting attractive but invalid conclusions.

This creates a new systems-engineering question: **What should an AI model do when one fast pass is not enough?**

One option is to train a larger model. Another is to let an existing model spend more work *while answering*: generate alternative solutions, revise a candidate, use a verifier, search a tree of partial solutions, execute tests, or allocate more compute only to cases that warrant it. This is the world of **reasoning models** and **test-time compute**.

It is also a world full of traps. Repeated sampling can produce many correlated mistakes. A reward model can favor a convincing but incorrect solution. Long deliberation can get stuck in circular analysis. An apparently impressive benchmark gain can disappear after accounting for latency, tool costs, and distribution shift. A production service cannot simply spend unlimited time on every problem.

This article develops the topic from first principles through practical deployment. We will separate established findings from assumptions, work through the mathematics, examine the major search strategies, build a runnable demonstration, and finish with a concrete production design.

**Reading path:** If you are new to reasoning models, start with Sections 1–6. If you care about research algorithms, focus on Sections 7–14. If you build systems, prioritize Sections 15–21.

---

## Table of Contents

1. [The three distinct meanings of AI thinking](#1-the-three-distinct-meanings-of-ai-thinking)
2. [From next-token prediction to problem solving](#2-from-next-token-prediction-to-problem-solving)
3. [Why reasoning models are different](#3-why-reasoning-models-are-different)
4. [Train-time scaling and test-time scaling](#4-train-time-scaling-and-test-time-scaling)
5. [The full reasoning request lifecycle](#5-the-full-reasoning-request-lifecycle)
6. [The mathematics of a compute budget](#6-the-mathematics-of-a-compute-budget)
7. [Best-of-N sampling and oracle upper bounds](#7-best-of-n-sampling-and-oracle-upper-bounds)
8. [Self-consistency and majority voting](#8-self-consistency-and-majority-voting)
9. [Verifiers: checking answers instead of trusting fluency](#9-verifiers-checking-answers-instead-of-trusting-fluency)
10. [Process supervision versus outcome supervision](#10-process-supervision-versus-outcome-supervision)
11. [Search over partial solutions](#11-search-over-partial-solutions)
12. [Reinforcement learning for reasoning](#12-reinforcement-learning-for-reasoning)
13. [Adaptive and compute-optimal reasoning](#13-adaptive-and-compute-optimal-reasoning)
14. [When additional thinking makes things worse](#14-when-additional-thinking-makes-things-worse)
15. [Runnable Python demonstration](#15-runnable-python-demonstration)
16. [Serving architecture and the GPU reality](#16-serving-architecture-and-the-gpu-reality)
17. [Evaluation: separating capability from selection](#17-evaluation-separating-capability-from-selection)
18. [Safety, security, and governance](#18-safety-security-and-governance)
19. [Economics and capacity planning](#19-economics-and-capacity-planning)
20. [A production implementation playbook](#20-a-production-implementation-playbook)
21. [Failure-mode troubleshooting guide](#21-failure-mode-troubleshooting-guide)
22. [Frequently asked technical questions](#22-frequently-asked-technical-questions)
23. [Final perspective](#23-final-perspective)
24. [Research references](#24-research-references)

---

## 1. The three distinct meanings of AI thinking

Before discussing architectures, we need to disambiguate three frequently conflated concepts.

| Concept | What changes? | Typical methods | Important distinction |
|---|---|---|---|
| **Reasoning-time generation** | How many tokens or internal steps a model generates in an inference request | Long-form deliberation, revision, decomposition | Usually **does not update weights** |
| **Inference-time search** | How many alternatives the system explores, scores, and compares | Best-of-N, voting, beam search, tree search, tool-based validation | Requires a way to **choose** among candidates |
| **Test-time training** | Model parameters or a task-specific adaptation state are updated from inference-time data | Online gradient steps, test-time adaptation | Is a form of **learning at test time**, not just longer generation |

All can consume more computation after a system has received a task, but they are not the same algorithm. This article focuses primarily on the first two, with a brief discussion of the third.

The phrase *thinking* is convenient shorthand for computation devoted to solving a problem. It should **not** be interpreted as proof that a model experiences human-like thought. It is also not safe to assume that a commercial reasoning model exposes its complete internal computation to the user or application.

### A useful mental model

Imagine solving a hard interview problem:

- **One-shot answer:** Write down the first approach that comes to mind.
- **Longer deliberation:** Work through alternatives, constraints, and counterexamples before answering.
- **Parallel search:** Ask several independent solvers to attempt the problem.
- **Verification:** Run tests, check a proof, or validate each step.
- **Adaptive compute:** Spend very little time on an obvious case, but more time on a difficult one.

Those are different ways to spend a limited problem-solving budget. The same distinction matters in AI.

---

## 2. From next-token prediction to problem solving

An autoregressive language model represents a sequence as a product of conditional token probabilities:

$$
P_{\theta}(x_{1:T})=\prod_{t=1}^{T}P_{\theta}(x_t\mid x_{<t}).
$$

During pretraining, a common objective is to minimize negative log-likelihood:

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
=-\mathbb{E}_{x\sim\mathcal{D}}\left[
  \sum_t\log P_{\theta}(x_t\mid x_{<t})
\right].
$$

This objective teaches the model to predict token sequences. Depending on the training data and subsequent post-training, it may acquire useful capabilities for mathematics, coding, inference, and planning.

But *predicting a likely continuation* and *guaranteeing a correct conclusion* are different objectives. A proof can sound plausible and still hide an invalid step. A program can look idiomatic and still fail edge cases. A numerical answer can be confidently wrong.

That distinction motivates three complementary components:

1. **Proposer:** Generates one or more candidate solutions.
2. **Verifier:** Estimates or determines whether a candidate satisfies the task.
3. **Controller:** Decides what to do next: answer, revise, branch, execute a test, ask for information, or stop.

The model may serve as more than one of these components, but they have different responsibilities and failure modes.

For verifiable tasks, such as code with executable tests or mathematical problems with computable checks, the separation is particularly powerful. For open-ended tasks, verification is much harder: the final answer may require expert judgment, external evidence, or a carefully designed rubric.

---

## 3. Why reasoning models are different

There is no single universal algorithm called a *reasoning model*. The term commonly refers to a model or system optimized to perform stronger multi-step problem solving, sometimes by using additional computation before providing a final response.

### 3.1 Structured intermediate computation

A system may produce intermediate work such as candidate plans, calculations, tests, or checks before returning a final answer. In some implementations this work appears in text; in others it is represented by hidden or non-user-facing computation. The important engineering property is **whether the additional computation improves verified outcomes**.

### 3.2 Learning to use additional computation

A model trained for short, direct answers is not automatically good at longer deliberation. It may repeat the same hypothesis or produce increasingly ornate explanations without correcting the underlying mistake.

Post-training can teach strategies such as checking conditions, exploring alternatives, using tools, and revising assumptions. Public work on OpenAI's o1 and DeepSeek-R1 helped demonstrate the practical relevance of reinforcement learning and extended reasoning behavior. Those results should be interpreted in their reported settings rather than as a guarantee that every extra token helps every task. See [OpenAI's reasoning report](https://openai.com/index/learning-to-reason-with-llms/) and [DeepSeek-R1's Nature paper](https://www.nature.com/articles/s41586-025-09422-z).

### 3.3 Reasoning strength has multiple dimensions

A useful taxonomy includes:

| Dimension | Question |
|---|---|
| **Decomposition** | Can the system break a problem into valid subproblems? |
| **Exploration** | Can it generate meaningfully different candidates? |
| **Verification** | Can it distinguish correct from plausible-but-wrong candidates? |
| **Revision** | Can it correct a detected mistake? |
| **Tool use** | Can it call calculators, code execution, search, or domain solvers appropriately? |
| **Calibration** | Does it know when evidence is insufficient? |
| **Stopping** | Can it stop when further computation is unlikely to help? |

A strong system may excel on some of these dimensions and remain weak on others. It is therefore misleading to reduce all reasoning quality to a single “thinking-token count.”

---

## 4. Train-time scaling and test-time scaling

Historically, much of the focus on stronger language models involved more training data, more parameters, or more training computation. Test-time scaling adds another axis: **spend additional computation for a particular question after the model has been trained**.

### 4.1 The trade-off

Suppose two deployment strategies are available:

- **Strategy A:** Use a larger model that answers in one pass.
- **Strategy B:** Use a smaller model that explores several candidates or verifies an extended attempt.

Strategy B may perform better on some tasks and worse on others. Its value depends on how hard the problem is, how diverse the candidate solutions are, whether selection is reliable, and how much extra computation is allowed.

The ICLR 2025 paper *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Parameters for Reasoning* systematically investigated this trade-off for math reasoning and found that compute allocation tailored to problem difficulty could be materially more efficient than a fixed best-of-N baseline in the studied setting. It is an **empirical, task-dependent result**, not a universal law. [Primary source](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html).

### 4.2 Why task difficulty matters

Consider three situations:

**Easy problem.** The base model is already very likely to be correct. Spending ten times more compute may yield negligible improvement.

**Intermediate problem.** The base model has a meaningful chance of producing a correct solution. Additional search and verification may discover and select one.

**Very hard problem.** The model may consistently misunderstand the task or lack necessary knowledge. More samples of the same flawed approach can waste resources. A larger or more specialized model, external tool, better evidence, or human assistance may be more effective.

This suggests that the optimal inference strategy is conditional, not fixed.

### 4.3 Test-time training is separate

Another research direction temporarily adapts model parameters using examples or self-supervision available during a test problem. This is not equivalent to simply generating more reasoning text. Research on test-time training for few-shot learning demonstrates such parameter adaptation in selected settings. [Source](https://proceedings.mlr.press/v267/akyurek25a.html).

Production adoption of test-time training involves additional concerns: isolation between requests, data privacy, adaptation stability, serving complexity, and the cost of updating parameters. Most practical reasoning-serving stacks do **not** need per-request weight updates to support inference-time search or verification.

---

## 5. The full reasoning request lifecycle

An end-to-end reasoning request can be modeled as a controlled search-and-verification workflow:

```text
User request
    |
    v
Input validation + task classification
    |
    v
Difficulty / risk / budget router
    |
    +-- Easy --> direct generation --> checks --> response
    |
    +-- Moderate --> sample / revise --> verifier --> response
    |
    +-- Difficult --> multiple candidates / search / tools
    |                                 |
    |                                 v
    |                          evidence / code tests
    |                                 |
    +--------------------------- verifier / judge
                                      |
                            +---------+---------+
                            |                   |
                        sufficient?          insufficient?
                            |                   |
                          respond        retry / escalate / stop
                            |
                            v
                  Log quality + cost + latency
```

Each stage has an independent engineering function:

- The **router** estimates difficulty and selects a budget.
- The **generator** proposes candidates.
- The **executor** performs allowed computations or tests.
- The **verifier** checks factual, formal, or task-specific correctness.
- The **controller** enforces maximum work, deadlines, and stop conditions.
- The **telemetry layer** records costs, success, and failure categories.

A good controller does not confuse “a response was generated” with “the answer was verified.”

---

## 6. The mathematics of a compute budget

Let $x$ be the question, $a$ a candidate answer, and $b$ an inference budget. The system chooses a policy $\pi$ that can generate, verify, and stop.

A simplified optimization objective is:

$$
\max_{\pi}\; \mathbb{E}\bigl[Q(x,a_{\pi})\bigr]
\quad\text{subject to}\quad
\mathbb{E}\bigl[C_{\pi}(x)\bigr]\leq B.
$$

Here:

- $Q$ represents a defined measure of answer quality or task utility.
- $C_{\pi}(x)$ is the cost of the chosen inference strategy.
- $B$ is the allowed budget.

A business-oriented version incorporates quality, latency, and monetary cost:

$$
U(\pi\mid x)=
\mathbb{E}[V\cdot\mathbf{1}\{\text{success}\}]
-\lambda_c\,\mathbb{E}[C_{\$}]
-\lambda_l\,\mathbb{E}[L]
-\lambda_r\,\mathbb{E}[R].
$$

$V$ is the value of a successful outcome; $C_{\$}$ is monetary cost; $L$ is latency; $R$ is a risk or failure penalty; and the $\lambda$ terms translate competing objectives into a common utility scale. This is a **decision model**, not a measured universal equation.

### 6.1 The budget is not just tokens

Inference-time work may include:

$$
C_{\mathrm{total}}=
C_{\mathrm{generation}}+
C_{\mathrm{verification}}+
C_{\mathrm{tools}}+
C_{\mathrm{coordination}}.
$$

Some organizations measure GPU-seconds or FLOPs; others measure token charges, API requests, tool execution, or total dollars. These are not interchangeable units, so use a consistent accounting system when optimizing.

### 6.2 Marginal return is the key

Increasing inference budget from $B$ to $B+\Delta B$ should be justified by its *incremental* improvement:

$$
\frac{\Delta\,\mathbb{E}[Q]}{\Delta B}.
$$

If this marginal gain becomes too small relative to the additional cost or deadline pressure, the system should stop. On many tasks, the response curve exhibits diminishing returns; on some tasks it can even degrade.

---

## 7. Best-of-N sampling and oracle upper bounds

The simplest parallel search strategy generates $N$ candidate solutions and selects one.

```text
Problem -> Candidate 1 --\
        -> Candidate 2 ---+--> Verifier --> Selected answer
        -> Candidate 3 --/
```

A crucial distinction is whether *a correct candidate exists* and whether *the selector actually chooses it*.

### 7.1 Probability that at least one candidate is correct

Assume each candidate is independently correct with probability $p$, and the value of $p$ stays constant as we sample. Then:

$$
P(\text{at least one correct among }N)
=1-(1-p)^N.
$$

For an **illustrative** $p=0.40$ and $N=5$:

$$
1-(1-0.40)^5=1-0.6^5=0.92224.
$$

That is a 92.224% chance that **at least one** of the five samples is correct—not a 92.224% chance that the system returns the correct answer.

The latter would require an ideal selector capable of finding the correct candidate. Real verifiers make errors; samples can be correlated; and $p$ can change across prompts and sampling settings. Therefore, the result is an idealized **oracle pass@N bound**, not a deployment forecast.

### 7.2 Why independence matters

Suppose each sample repeats the same fundamental misconception. Adding more samples will not provide the diversity assumed by the formula. In the extreme case of perfectly correlated success/failure, the probability of finding a correct candidate remains $p$, regardless of $N$.

Sampling diversity, prompt design, model diversity, and distinct tool evidence can matter as much as raw sample count.

### 7.3 How a verifier changes the outcome

Define a scoring function $s(x,a)$ and choose:

$$
\hat a=\operatorname*{arg\,max}_{a_i\in\{a_1,\ldots,a_N\}}s(x,a_i).
$$

This selection procedure is only useful to the extent that scores discriminate true correctness on the target distribution. If a verifier rewards fluent explanations over valid logic, increasing $N$ can increase the chance of finding a confidently wrong answer with an unusually high score.

**Best-of-N is a generation method plus a selection problem. Optimize both.**

---

## 8. Self-consistency and majority voting

Self-consistency generates multiple reasoning trajectories, extracts their final answers, and chooses a commonly occurring answer rather than relying on one trajectory.

For answers $a_1,\ldots,a_N$, majority voting can be written as:

$$
\hat a=\operatorname*{arg\,max}_{y}
\sum_{i=1}^{N}\mathbf{1}[a_i=y].
$$

For example, five candidates might return:

| Candidate | Final answer |
|---|---|
| 1 | 42 |
| 2 | 45 |
| 3 | 42 |
| 4 | 42 |
| 5 | 45 |

The voted answer is 42. But 42 could still be wrong. **Consensus measures agreement, not ground-truth correctness.**

Self-consistency is attractive when final answers are easily canonicalized and different valid reasoning paths should converge. It is harder for ambiguous natural-language answers, tasks with multiple correct outputs, or problems where errors share a common source.

Research from Wang and colleagues found improvements across several reasoning benchmarks using self-consistency sampling. These are benchmark results, not a guarantee that voting dominates every other strategy. [Paper](https://arxiv.org/abs/2203.11171).

### When voting is especially fragile

- Candidates copy the same false assumption.
- The answer parser merges distinct results incorrectly.
- A common wrong answer is more frequent than a rarer correct one.
- The task has valid answers that are semantically equivalent but formatted differently.
- Sampling temperature or prompt design reduces meaningful diversity.

Where an external checker exists, combine voting with verification rather than treating consensus as proof.

---

## 9. Verifiers: checking answers instead of trusting fluency

A **verifier** estimates or determines whether a candidate satisfies the relevant task requirements. The strongest type of verifier depends on the domain.

| Task | Candidate verifier | Main limitation |
|---|---|---|
| Arithmetic | Exact calculation, symbolic algebra | Requires correct problem formalization |
| Code | Unit tests, property tests, type checks | Tests may be incomplete or contaminated |
| SQL | Parser, schema validation, read-only execution | Valid syntax does not ensure correct meaning |
| Scientific answer | Evidence check and expert review | Evidence may be incomplete or contested |
| Formal proof | Proof assistant / formal checker | Translating a claim into formal syntax is hard |
| Open-ended writing | Rubric, learned judge, human review | Subjective and model-dependent |

### 9.1 Deterministic checkers are special

If a mathematics result can be checked by exact arithmetic or a theorem prover, the verifier can be much more trustworthy than a second language model making a subjective judgment. Nevertheless, correctness of a formal check depends on encoding the right question and assumptions.

For programming, test suites provide executable evidence, but passing the supplied tests is not equivalent to being correct for all possible inputs.

### 9.2 Learned verifiers are useful but fallible

A learned verifier may assign a score:

$$
s_\phi(x,a)\approx P(\text{correct}\mid x,a).
$$

Unless the score has been properly calibrated on representative data, it should not be interpreted as an actual probability. Even a calibrated model can fail under distribution shift.

One common challenge is **selection-induced overoptimization**. If we draw many candidates and always select the highest scored, the chosen sample may exploit quirks in the verifier rather than be genuinely correct. This is a form of Goodhart's law: optimizing a proxy score can weaken its relationship with the real objective.

### 9.3 Independent evidence matters

If the proposer and verifier share the same blind spots, their agreement can be misleading. Independence can be improved by using a different verification method, deterministic tool results, separate evidence, or qualified human review.

Cobbe and colleagues studied training learned verifiers to assess generated mathematical solutions, establishing an important line of work on the proposer/verifier separation. [Paper](https://arxiv.org/abs/2110.14168).

---

## 10. Process supervision versus outcome supervision

Not all feedback arrives at the same level of detail.

### 10.1 Outcome supervision

An **outcome reward model (ORM)** scores an overall answer or trajectory based on final correctness or success. It is comparatively straightforward when a reliable final checker exists.

For example:

- Did the code pass the tests?
- Was the numerical result correct?
- Did the theorem statement meet the final specification?

Outcome labels do not necessarily explain *where* a long solution first went wrong.

### 10.2 Process supervision

A **process reward model (PRM)** scores intermediate steps rather than only the final result. If a candidate contains steps $z_1,\ldots,z_T$, a process evaluator can assign signals to individual transitions.

A generic example of aggregating step quality is:

$$
S_{\mathrm{process}}(z_{1:T})=
\sum_{t=1}^{T}w_t\,r_\phi(z_t\mid x,z_{<t}).
$$

This is an **illustrative aggregation**; different methods use different aggregations, including minima or products of step scores. It is not a universal PRM formula.

Process feedback offers potentially finer-grained credit assignment, but it is harder and more expensive to label correctly. Some intermediate steps are ambiguous or depend on later steps, and learned scores can still be misled.

The research paper *Let's Verify Step by Step* compared process and outcome supervision for mathematical reasoning and released the PRM800K dataset. Its conclusions are specific to its training and evaluation setup. [ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html).

### 10.3 A concrete example

Suppose a model solves a discounted cash-flow problem and obtains the wrong answer. An outcome checker can tell us the total value is incorrect. A process checker might detect that the discount rate was entered as `10` instead of `0.10`, preventing downstream calculations from being trusted.

The practical benefit is not more verbose reasoning; it is **localizing errors before they contaminate later decisions**.

---

## 11. Search over partial solutions

Best-of-N generates whole candidate answers. More structured search operates over **partial solutions**.

### 11.1 Breadth-first and beam-style exploration

At each step, the system proposes multiple continuations, scores them, and retains a set of promising states. A beam of width $b$ retains only $b$ candidates per expansion step.

This reduces the search space but risks pruning the correct path if early scores are misleading.

### 11.2 Tree search

A tree search explores branches representing decisions, intermediate deductions, or executable actions. A controller chooses which nodes to expand based on scores, visit counts, or exploration policies.

A common Monte Carlo Tree Search exploration rule is UCB-inspired:

$$
\operatorname{UCB}(i)=\bar{r}_i+
 c\sqrt{\frac{\ln N}{n_i}},
$$

where $\bar r_i$ is an estimated reward, $N$ is the total number of parent visits, $n_i$ is visits to child $i$, and $c$ controls exploration. This is a **generic bandit-style expression**, not a claim that all language-model tree search systems use precisely this scoring rule.

### 11.3 Search changes what the verifier must do

Whole-answer selection asks which finished answer is best. Tree search asks whether a *partial state is promising enough to continue*. Predicting the value of an unfinished reasoning path can be substantially harder than checking a completed numerical result.

### 11.4 When search is worth its overhead

Search is a good candidate when:

- The problem naturally decomposes into decisions or subgoals.
- Intermediate states can be evaluated meaningfully.
- Alternative paths have materially different outcomes.
- The cost of a wrong decision justifies additional computation.

It can be wasteful when the problem is short, verifiers are weak, or the task lacks a useful state representation.

For comparative research on greedy decoding, best-of-N, majority voting, and tree search under compute budgets, see the ICLR 2025 paper on inference scaling laws: [Wu et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c3caae2f725c8e2a55ecd600563d172-Abstract-Conference.html).

---

## 12. Reinforcement learning for reasoning

A generator can also be **trained** to produce more useful reasoning behaviors rather than relying exclusively on external inference-time search.

### 12.1 Policy optimization

Treat the model as a policy $\pi_\theta(y\mid x)$ that generates output trajectories $y$ for prompts $x$. A simplified objective is to maximize expected reward:

$$
J(\theta)=
\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}
[R(x,y)].
$$

In practice, algorithms often include regularization to prevent destructive drift. An illustrative KL-regularized objective is:

$$
J_{\mathrm{KL}}(\theta)=
\mathbb{E}[R(x,y)]-
\beta\,\mathbb{E}_{x}
\left[D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot\mid x)
\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\bigr)\right].
$$

This captures a broad policy-optimization idea; it is **not** a faithful specification of any particular proprietary training recipe or a complete description of DeepSeek-R1's actual optimizer.

### 12.2 The reward-design problem

Rewards must be tied to the desired outcome. If a model receives positive feedback simply for producing long explanations, it may learn verbosity instead of correctness. If it is rewarded for passing a weak test suite, it may learn to exploit gaps in the tests.

For verifiable domains, deterministic correctness signals are attractive. For open-ended tasks, reward quality depends on evidence, rubric design, human feedback, or learned judges.

### 12.3 What public reasoning-model research shows

DeepSeek-R1's published research described reinforcement-learning approaches that incentivize extended reasoning behavior and differentiate an RL-focused model from a multi-stage training pipeline. The published experiments are informative but do not imply that all models must use the same method. [Nature research article](https://www.nature.com/articles/s41586-025-09422-z).

OpenAI's 2024 discussion of o1 reported that both reinforcement-learning training compute and test-time compute were relevant to difficult reasoning benchmarks. [Source](https://openai.com/index/learning-to-reason-with-llms/).

Research on **s1: Simple Test-Time Scaling** further examined fine-tuning on a selected collection of reasoning examples combined with explicit reasoning-budget control. The method illustrates that inference-time behavior depends heavily on how the model is trained to use a budget. [EMNLP paper](https://aclanthology.org/2025.emnlp-main.1025/).

### 12.4 Why RL is not equivalent to inference search

Reinforcement learning updates the policy so it learns better actions across tasks. Inference-time search spends computation on the current task. They can complement each other: a better-trained proposer may generate higher-quality candidates, while better search and verification may extract more value from a fixed proposer.

---

## 13. Adaptive and compute-optimal reasoning

A fixed budget—say, ten candidate samples for every problem—is simple to implement but frequently inefficient.

A more practical controller should select the inference strategy based on estimated difficulty, consequence, and available resources.

### 13.1 Budget allocation as a decision problem

For each task $x$, select a strategy $s$ and budget $b$:

$$
(s^*,b^*)=
\operatorname*{arg\,max}_{s,b}
\left[\mathbb{E}[Q\mid x,s,b]
-\lambda\,C(x,s,b)\right].
$$

This expression describes the objective, not an algorithm that automatically gives a correct router. The system needs data to estimate quality and cost as functions of task type and budget.

### 13.2 Strategies an adaptive router can choose

| Task signal | Possible strategy | Why |
|---|---|---|
| Straightforward extraction with reliable source | Direct answer + schema check | Avoids needless search |
| Medium-difficulty arithmetic | Candidate + exact calculation check | Strong checker can catch errors cheaply |
| Coding task with tests | Generate, execute sandboxed tests, repair | Executable feedback is informative |
| Hard math with uncertain answer | Diverse candidates + verifier / search | Exploration can uncover valid alternatives |
| Missing authoritative evidence | Retrieve sources or abstain | Longer text cannot create missing facts |
| High-impact action | Tool validation + human approval | Consequence dominates speed |

### 13.3 Value of computation

Suppose the current best answer has estimated value $V_0$, and another unit of computation is expected to raise it to $V_1$. The expected value of computation is:

$$
\operatorname{VOC}=\mathbb{E}[V_1-V_0]
-\operatorname{Cost}(\text{next action}).
$$

A rational controller continues only when the expected benefit is positive and hard safety/deadline constraints remain satisfied. Estimating that benefit correctly is itself a nontrivial ML problem.

### 13.4 Calibrating a difficulty router

A difficulty score is useful only if it predicts actual benefit from additional compute. Features can include input length, domain, retrieved evidence quality, prior uncertainty estimates, and results of cheap initial checks. Avoid allowing a cheap model's self-reported confidence to be the only signal.

Evaluate routing decisions with held-out tasks. Compare the router to simple baselines: fixed budget, direct-only, best-of-N, and always-expensive. Track both aggregate utility and results on high-risk subgroups.

---

## 14. When additional thinking makes things worse

It is tempting to expect a monotonic relationship: more computation → better answer. That expectation is unsafe.

### 14.1 Correlated failures

Multiple samples can share the same hidden assumption or training-data bias. More output does not guarantee more independent evidence.

### 14.2 Verifier overoptimization

With enough candidate samples, a system may find one that scores exceptionally well under a flawed verifier yet is incorrect. This is especially dangerous when the verifier is trained on the same distribution or uses the same model family as the generator.

### 14.3 Unproductive deliberation

A model may revise a correct answer into an incorrect one, chase irrelevant alternatives, or loop through superficial self-checks. Some experimental research has examined cases where added test-time compute is not beneficial; the effect depends on method and task rather than the number of tokens alone.

### 14.4 Latency and queuing costs

On a shared GPU, a long reasoning request can hold decoding resources and KV-cache capacity while other users wait. Increasing individual-task accuracy can worsen the service's overall p95/p99 latency, admission rate, or total successful requests per minute.

### 14.5 Tools can make false certainty worse

A calculator can compute exactly, but not determine whether a formula encodes the correct business requirement. A test suite can pass while missing critical edge cases. Retrieval can return an authoritative-looking but outdated policy.

**The rule is simple: spending more compute should add information, diversity, or verification—not merely more prose.**

---

## 15. Runnable Python demonstration

To make the mathematics tangible, the companion repository code implements a small, dependency-free simulation. It does **not** pretend to be a real LLM. Instead, it isolates the logic of candidate generation, oracle upper bounds, verifier quality, and adaptive sample allocation so each assumption can be inspected.

The full script is available in the companion code path:

[`examples/reasoning_test_time_compute_demo.py`]({{ '/examples/reasoning_test_time_compute_demo.py' | relative_url }})
Run it locally with Python 3.10 or later:

```bash
python3 examples/reasoning_test_time_compute_demo.py
```

### 15.1 What the simulator represents

For each task, a hypothetical proposer produces candidates. Each candidate is either correct or incorrect according to a simulated ground truth. A synthetic verifier assigns a score using separate distributions for correct and incorrect candidates.

We compare:

1. One sample.
2. Oracle pass@N: at least one correct sample exists.
3. Best-of-N with an imperfect score-based verifier.
4. A difficulty-aware policy allocating different sample counts.

### 15.2 What the simulator deliberately does not claim

- It does not benchmark a named reasoning model.
- It does not simulate actual token distributions or GPU execution.
- It assumes a particular probability model for illustrative purposes.
- It does not establish universal accuracy improvements or costs.
- It does not let a verifier inspect hidden ground truth when choosing candidates.

These limitations are important: a toy simulation clarifies the selection problem but cannot substitute for evaluation on real tasks.

### 15.3 Core selection code

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Candidate:
    is_correct: bool
    verifier_score: float


def select_best(candidates: list[Candidate]) -> Candidate:
    # The decision uses only verifier scores, never hidden correctness labels.
    if not candidates:
        raise ValueError("No candidates to select")
    return max(candidates, key=lambda item: item.verifier_score)


def oracle_has_solution(candidates: list[Candidate]) -> bool:
    # Evaluation-only diagnostic; not a deployable selection strategy.
    return any(item.is_correct for item in candidates)
```

The contrast between `select_best` and `oracle_has_solution` is the whole point: **finding a correct candidate is not the same thing as selecting it**.

---

## 16. Serving architecture and the GPU reality

Reasoning models do not run in an economic vacuum. They compete for GPU compute, memory bandwidth, KV-cache capacity, and request slots.

### 16.1 Longer answers create longer decode workloads

A causal language model typically generates output tokens sequentially. Longer reasoning traces often extend the decode phase, increasing time spent holding a request's KV cache and consuming decoding iterations.

This interacts with continuous batching, admission control, priority scheduling, and the distribution of active sequence lengths. Even if the hardware is efficient at generating tokens, an unbounded reasoning budget can degrade tail latency.

### 16.2 Parallel candidates are not free

Generating $N$ candidate outputs can use batching or multiple GPUs, but consumes additional aggregate model work. A rough cost model is:

$$
C_{\mathrm{request}}\approx
N\cdot C_{\mathrm{candidate}}
+C_{\mathrm{verifier}}+C_{\mathrm{tools}}.
$$

This approximation ignores shared-prefill benefits, variable lengths, caching, and other implementation details. **Latency is not necessarily $N$ times higher** if candidates are processed concurrently, but total compute and memory demand generally increase.

### 16.3 The standard service metrics still matter

| Metric | Interpretation for reasoning systems |
|---|---|
| **TTFT** | Delay before first user-visible output; may exclude hidden deliberation or include it depending on API semantics |
| **End-to-end latency** | Time until the final verified result is delivered |
| **TPOT / inter-token latency** | Streaming responsiveness during visible generation |
| **Throughput** | Work completed per unit time |
| **Goodput** | Requests completed successfully within the applicable SLO |
| **p95/p99 latency** | Slow-tail experience, especially under mixed difficulty |
| **Verifier cost** | Extra resources spent on selecting or checking candidates |
| **Cost per successful task** | Full application cost divided by verified successful completions |

Do not report TTFT without defining what counts as the first token; reasoning APIs can differ in whether intermediate reasoning is surfaced or accounted for separately.

### 16.4 Architecture for a production reasoner

A practical serving system separates the fast path from the expensive path:

```text
API / authentication
        |
        v
Request classifier -----> Risk policy
        |                     |
        v                     v
Budget planner --------> Hard limits / deadlines
        |
        v
Inference gateway
  |           |
  |           +--> Candidate sampler / search worker
  v                      |
Fast answer              v
  |                 Verifier / sandbox / evidence
  |                      |
  +-----------+----------+
              |
              v
        Decision controller
              |
      +-------+-------+
      |               |
    return          retry, escalate,
    result          or abstain
              |
              v
Telemetry: tokens, GPU time, errors, quality, cost
```

The inference gateway should enforce resource ceilings independently of the model's own preference for more reasoning.

---

## 17. Evaluation: separating capability from selection

Reasoning evaluations are particularly vulnerable to metric confusion. A system may show a strong pass@N oracle upper bound while having poor pass@1 or poor selected-answer accuracy.

### 17.1 Minimum evaluation matrix

| Metric | What it answers | Common interpretation error |
|---|---|---|
| **pass@1 / single-sample accuracy** | How often one candidate is correct | Hiding sampling settings |
| **Oracle pass@N** | Does *any* candidate work? | Presenting it as system success |
| **Selected-answer accuracy** | Does the actual selector choose correctly? | Ignoring verifier mistakes |
| **Success at fixed dollars** | What quality is obtained for a given price? | Comparing runs with unequal budgets |
| **Success at fixed latency** | What quality meets the deadline? | Ignoring timeouts and queue delays |
| **Calibration** | Do confidence or verifier scores match empirical correctness? | Treating arbitrary scores as probabilities |
| **Failure severity** | How costly are bad decisions? | Counting all errors equally |

### 17.2 Compare on a quality-cost frontier

A responsible experiment should compare configurations under matched constraints. A larger model, a smaller model with more samples, a smaller model plus a verifier, and a search-based method should be evaluated on the same tasks with carefully reported compute and latency.

A configuration is Pareto-dominated if another configuration achieves at least the same quality for lower cost (or at least the same quality and cost with lower latency), under the specified dimensions.

### 17.3 Temporal holdouts and contamination

Public math and coding benchmarks may appear in training data, prompt collections, or model-selection loops. A system may improve its score without improving its ability to solve new problems.

Use time-separated evaluation where feasible, assess benchmark leakage risk, and include fresh problems. LiveCodeBench was designed to collect new programming problems over time and assess multiple code-related abilities; MathArena similarly emphasizes new competition problems and contamination-aware evaluation. [LiveCodeBench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/94074dd5a072d28ff75a76dabed43767-Abstract-Conference.html) · [MathArena](https://proceedings.nips.cc/paper_files/paper/2025/hash/1d27c01ebd3e3aebe226b44fc970d803-Abstract-Datasets_and_Benchmarks_Track.html).

### 17.4 Run a meaningful ablation study

At minimum, report:

- Direct generation (one sample).
- Fixed best-of-N without a learned verifier (for example, voting where appropriate).
- Best-of-N with your selected verifier.
- Adaptive compute routing.
- The same methods with comparable total budgets.

Stratify results by difficulty, input domain, output length, and availability of external checks. Without these comparisons, it is difficult to know whether a gain comes from the base model, extra samples, verification, or simply more budget.

---

## 18. Safety, security, and governance

Reasoning improves some capabilities, but it does not remove the need for external controls.

### 18.1 A coherent answer is not authorization

If a reasoning system is connected to real-world tools, a long and persuasive justification must not grant it privileges. The tool layer should enforce permissions, allowlists, rate limits, validation, and approval requirements independently of the model.

### 18.2 Protect verifiers and sandboxes

Executable validation is powerful, but running generated code or queries carries risks. Use isolated execution environments, restrictive networking, bounded resources, input sanitation, and explicit permissions. A code test should never have unrestricted access to production credentials or customer data.

### 18.3 Avoid trusting self-reported confidence

A model may be able to describe its uncertainty, but natural-language confidence is not automatically calibrated. When risk matters, use task-grounded checks, external evidence, statistically validated scores, or qualified human review.

### 18.4 Make stopping and escalation explicit

The system should stop when:

- It exceeds a budget or deadline.
- It detects missing authoritative information.
- The best candidates conflict without a resolution method.
- Verification fails repeatedly.
- The next action would exceed its permissions.
- The cost or risk of further exploration is unjustified.

An honest abstention or escalation can be more valuable than a polished but unreliable final answer.

---

## 19. Economics and capacity planning

The most useful unit is often **cost per verified successful task**, not raw tokens per second or average model-call price.

$$
\operatorname{CPST}
=\frac{\text{Total measured operating cost}}
{\text{Number of verified successful tasks}}.
$$

Operating cost should include the boundaries relevant to the business: generation, verification, tools, retries, infrastructure, human review, and failures requiring rework.

### 19.1 A simple illustrative comparison

Assume two hypothetical workflows process 1,000 comparable tasks:

| Measure | Direct answer | Reasoning + verifier |
|---|---:|---:|
| Attempts | 1,000 | 1,000 |
| Verified successful tasks | 700 | 850 |
| Total operating cost | $100 | $170 |
| Cost / successful task | **$0.143** | **$0.200** |

The enhanced workflow raises accuracy but is **more expensive per successful task** in this toy scenario. It might still be valuable when errors are costly, but the product decision requires a value or penalty model.

Now suppose a verified success creates $1 of value and an unsuccessful task creates no value. The simplified expected net value is:

- Direct: $700 − $100 = $600.
- Reasoning + verifier: $850 − $170 = $680.

In this particular **illustrative** setup, the more expensive system creates $80 more net value, even though its cost per success is higher. Different success values, error penalties, and volume assumptions could reverse the choice.

### 19.2 Queueing changes economics

Peak throughput and average latency are not enough for a shared service. Long-running reasoning requests can increase queueing delay for shorter requests, reducing goodput under deadlines. Plan for mixed workloads, tail latency, timeouts, and graceful degradation.

### 19.3 Spend selectively

A practical cost-control strategy is to:

1. Attempt a cheap, well-evaluated fast path.
2. Run inexpensive deterministic checks.
3. Expand to candidate search only when those signals justify it.
4. Stop when marginal benefit becomes too small.
5. Escalate genuinely ambiguous, expensive, or high-risk cases.

This is the operational expression of compute-optimal reasoning.

---

## 20. A production implementation playbook

### Phase A — Define the task and success contract

Specify exactly what a correct answer means. For code, is it passing a hidden test suite? For analysis, is it agreeing with an expert rubric? For a business workflow, is it completing a verified external action? Define unacceptable failure modes separately from average accuracy.

### Phase B — Establish baselines

Collect a representative holdout set and measure direct-answer success, latency, cost, and confidence calibration. Record performance by difficulty rather than relying only on aggregate averages.

### Phase C — Add the cheapest useful verifier

Prefer exact or executable checks where they are valid: parsers, schema validators, symbolic mathematics, unit tests, or deterministic business rules. Avoid introducing an additional LLM judge when a stronger non-ML check is available.

### Phase D — Measure search benefit

Compare fixed sample counts (for example 1, 2, 4, and 8) at equal tasks and stated compute budgets. Report oracle pass@N separately from selected-answer accuracy.

### Phase E — Train or calibrate the router

Estimate whether extra computation is likely to improve a task. Test whether the router beats a fixed-budget baseline. Keep a fallback policy for out-of-distribution tasks.

### Phase F — Instrument serving costs

Log final success, model versions, generation/verification tokens, tool durations, GPU or API usage, queue time, verifier failures, and stopping reasons. Do not store sensitive deliberation or private data unnecessarily.

### Phase G — Test adverse conditions

Inject timeouts, corrupted tool responses, missing evidence, contradictory candidates, and malicious inputs. Confirm that the controller respects deadlines and access-control policies.

### Phase H — Roll out progressively

Start with shadow evaluation or a limited canary. Track quality-cost-frontier changes, user experience, and high-severity failures. Maintain the ability to roll back routing or verification changes.

### Phase I — Re-evaluate continuously

Data distributions, tools, models, and cost schedules change. Update benchmarks and task mixes accordingly. A compute allocation policy that was optimal for last quarter's workload may not be optimal today.

---

## 21. Failure-mode troubleshooting guide

| Symptom | Likely cause | Investigation | Mitigation |
|---|---|---|---|
| More samples but little accuracy gain | Correlated candidate failures | Measure answer diversity and oracle pass@N | Vary proposer or strategy; improve evidence |
| Oracle pass@N rises but final accuracy stalls | Weak candidate selector | Compare rank scores to ground truth | Improve verifier and calibration |
| Quality drops at very high budgets | Overthinking or proxy overoptimization | Plot quality versus budget | Bound search; add stop rules |
| Tail latency spikes | Long decode and queue contention | Inspect p95/p99 by task and output length | Separate traffic classes; cap budgets |
| Incorrect code passes tests | Incomplete test suite | Mutation tests and hidden cases | Add property tests and independent checks |
| “Confidence” is high on failures | Miscalibration | Reliability diagrams / error stratification | Use evidence-based or calibrated decisions |
| Search tree explodes | Excessive branching | Track expansions and verifier calls | Pruning, beam limits, early stopping |
| Cost per success worsens | Gains below marginal cost | Compare frontier and task economics | Adaptive routing; simplify easy path |
| Tools leak permissions | Unsafe execution layer | Audit policies and sandbox boundary | Enforce least privilege outside model |
| Results collapse on fresh tasks | Contamination or distribution shift | Temporal holdout and drift audit | Refresh evaluation and training strategy |

---

## 22. Frequently asked technical questions

### Is a reasoning model just a larger language model?

Not necessarily. Reasoning ability depends on architecture, pretraining, post-training, prompts, tool access, available inference compute, and verification strategy. Model parameter count is only one factor.

### Does a longer chain of thought guarantee a better answer?

No. Length is a resource measure, not a correctness proof. The model can repeat mistakes, make new ones during revision, or consume compute without acquiring new information.

### Is best-of-N equivalent to majority voting?

No. Best-of-N chooses a candidate using a scoring function. Majority voting chooses the most frequent extracted answer. Both use multiple candidates, but their selectors and assumptions differ.

### What is the difference between pass@N and selected-answer accuracy?

Oracle pass@N asks whether at least one candidate is correct. Selected-answer accuracy asks whether the actual deployed selector returned a correct answer. The difference is the verifier/selection gap.

### Does reinforcement learning happen during each request?

Usually not. In standard serving, RL has already modified the model during training. Inference-time search uses the resulting policy without changing weights. Test-time training is a separate family of methods that does perform adaptation during inference.

### When should I use a deterministic verifier?

Whenever a reliable formal or executable check exists for the requirement. Examples include syntax validation, exact arithmetic, type checking, and bounded sandboxed tests. Remember that an exact checker can still verify the wrong specification.

### Should I always choose a smaller model with more thinking time?

No. On some tasks, the larger model may have capabilities or domain knowledge absent from the smaller model. Compare configurations under matched budget, latency, and success criteria.

### Why might a verifier get worse as I increase samples?

Selection over more samples can expose a weak learned verifier to increasingly unusual candidates that exploit its scoring errors. Evaluate the entire selection process at the actual sample counts used in production.

### How is this different from agent architecture?

A reasoning controller allocates compute to solve a cognitive task. An agent architecture additionally manages tools, state, authorization, persistence, execution, recovery, and verification of external outcomes. These topics intersect, but neither replaces the other.

### What should Staff/Principal engineers focus on?

Own the end-to-end objective: measurement design, matched compute experiments, workload-aware routing, independent verification, infrastructure bottlenecks, failure budgets, security boundaries, and cost per reliable outcome. A novel search algorithm is valuable only if it improves the actual product frontier.

---

## 23. Final perspective

The next generation of AI capability is not necessarily a choice between training a bigger model and deploying a cheaper one. It is about **how intelligence is allocated**.

Some tasks deserve one quick response. Others deserve several competing hypotheses, carefully chosen tools, executable tests, and independent checks. The hardest tasks may still require expert intervention or a fundamentally more capable model.

Reasoning research provides a growing set of strategies for spending additional computation: longer deliberation, parallel sampling, self-consistency, verifiers, tree search, reinforcement learning, and adaptive routing. But their value depends on the question, the quality of the checker, and the constraints of the system delivering the answer.

The most important distinction is not **fast versus slow**. It is **productive computation versus unproductive computation**.

A trustworthy reasoning system should know what it is optimizing, explore when exploration has value, verify whenever evidence permits, stop when uncertainty cannot be resolved economically, and measure success in the real application rather than through impressive-looking reasoning traces.

> **The future of reasoning AI will not be defined merely by how long a model can think. It will be defined by how effectively a system turns additional computation into better verified decisions.**

**Discussion:** In your work, where has added test-time compute helped most—coding, mathematical problem solving, scientific analysis, or agent planning? And when has it made the system slower without meaningfully improving quality?

---

## 24. Research references

The following primary sources informed this article. Findings have been paraphrased, and published benchmark results have not been treated as universal guarantees.

1. **OpenAI (2024).** *Learning to Reason with LLMs.* [Research announcement](https://openai.com/index/learning-to-reason-with-llms/). Discusses o1, reinforcement learning, and observed test-time scaling.
2. **DeepSeek-AI et al. (2025).** *DeepSeek-R1 Incentivizes Reasoning in LLMs through Reinforcement Learning.* [Nature](https://www.nature.com/articles/s41586-025-09422-z) · [arXiv](https://arxiv.org/abs/2501.12948). Public model-training and reasoning research.
3. **Snell, Lee, Xu, and Kumar (2025).** *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Parameters for Reasoning.* [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2408.03314). Compute-optimal test-time reasoning on studied tasks.
4. **Wu et al. (2025).** *Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving.* [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c3caae2f725c8e2a55ecd600563d172-Abstract-Conference.html). Comparison of inference strategies and budgets.
5. **Wang et al. (2023).** *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* [arXiv](https://arxiv.org/abs/2203.11171) · [ICLR record](https://mlanthology.org/iclr/2023/wang2023iclr-selfconsistency/). Self-consistency and majority aggregation.
6. **Cobbe et al. (2021).** *Training Verifiers to Solve Math Word Problems.* [arXiv](https://arxiv.org/abs/2110.14168). Learning to rank and check mathematical solutions.
7. **Lightman et al. (2024).** *Let's Verify Step by Step.* [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html). Process versus outcome supervision and PRM800K.
8. **Muennighoff et al. (2025).** *s1: Simple Test-Time Scaling.* [EMNLP](https://aclanthology.org/2025.emnlp-main.1025/) · [arXiv](https://arxiv.org/abs/2501.19393). Budget control and supervised reasoning examples.
9. **Setlur et al. (2025).** *Scaling Test-Time Compute Without Verification or RL Is Suboptimal.* [ICML / PMLR](https://proceedings.mlr.press/v267/setlur25a.html). Theoretical comparisons involving verification and learning.
10. **Akyürek et al. (2025).** *The Surprising Effectiveness of Test-Time Training for Few-Shot Learning.* [ICML / PMLR](https://proceedings.mlr.press/v267/akyurek25a.html). Parameter adaptation at test time.
11. **Jain et al. (2025).** *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/94074dd5a072d28ff75a76dabed43767-Abstract-Conference.html). Fresh coding evaluations.
12. **MathArena (2025).** *Evaluating LLMs on Uncontaminated Math Competitions.* [NeurIPS](https://proceedings.nips.cc/paper_files/paper/2025/hash/1d27c01ebd3e3aebe226b44fc970d803-Abstract-Datasets_and_Benchmarks_Track.html). Temporal math evaluation and contamination concerns.

---

*Publication note: All numerical examples labeled illustrative are constructed teaching examples, not measurements from a deployed model. Claims about specific research papers should be interpreted within their reported experimental settings. Last reviewed: November 25, 2026.*