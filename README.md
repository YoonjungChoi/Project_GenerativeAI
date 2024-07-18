# Generative AI 

**Compute resources**
Full-tuning requires lots of compute resources; not only memory but also various other parameters that are required during the training process. 
In contrast to full-tuning, PEFT provcies a set of techiniques allowing you to fine-tune your models while utilizing less compute resources.

**catastrophic forgetting**
fine-tuning one specific task impact on LLM degrading on other tasks. To avoid degradition, we can train one specific task with all other tasks together.

# 1. PEFT Parameters Effective Fine Tuning



## 1.1. Adapters (LoRA is the most commonly used amongst them)

the two low-rank matrices are multiplied together to create a matrix with the same dimensions as the frozen weights. You then add this to the original weights and replace them in the model with these updated values. You now have a LoRA fine-tuned model that can carry out your specific task. Because this model has the same number of parameters as the original, there is little to no impact on inference latency. Researchers have found that applying LoRA to just the self-attention layers of the model is often enough to fine-tune for a task and achieve performance gains. 

![image](https://github.com/user-attachments/assets/a5d17389-2593-4ee5-af84-2e4fbfc3f15e)

**QLoRA**

QLoRA is a fine-tuning method that combines Quantization(in the context of deep learning is the process of reducing the numerical precision of a model's tensors, making the model more compact and the operations faster in execution) and Low-Rank Adapters (LoRA). QLoRA is revolutionary in that it democratizes fine-tuning: it enables one to fine-tune massive models with billions of parameters on relatively small, highly available GPUs. QLoRA aims to futher reduce memory requirements by combining low-rank adaptation with quatization. 

```
bnb_config = BitAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="n4f",
  bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_checkpoint, quatization_config=bnb_config)
```

## 1.2. Soft-Prompts. (Prompt Tuning)
With LoRA, the goal was to find an efficient way to update the weights of the model without having to train every single parameter again. There are also additive methods within PEFT that aim to improve model performance without changing the weights at all.

[Prompt-Tuning HuggingFace Doc](https://huggingface.co/learn/cookbook/prompt_tuning_peft)

# 2. RLHF Reinforcement Learning Human Feedack

Human Alignment: Helpful, Honest, and Harmless. **HHH**

RL is consist of Agent, Policy, Action, Rewards.

**Collecting a Custom Dataset**

To collect Training Dataset with Human-in-the-Loop, instruction should clearly descrie the task for the labler. Providing these detailed human instructions will increase the likelihood that the responses will be high quality and that all individual humans will carry out the laeling task in a consistent way. Using Amazon SageMaker Ground Truth for human annotations.

![image](https://github.com/user-attachments/assets/4add3685-64c9-4185-b9bc-511d297e4dfc)

**Using the Reward Model with RLHF**
RLHF is a fine-tuning process that modifies the underlying weights of a given generative model to etter align with the human preferences expressed through the reward model. The reward model captures human preferences through direct human feedback using service like SageMaker GroundTruth.

The model may generate completion and reward model produce a positive/negarive reward value for the completion.

![image](https://github.com/user-attachments/assets/3750b875-2b9a-464c-8bc9-1b329b7a76a7)
![image](https://github.com/user-attachments/assets/a20feb34-d8f8-4c19-9253-5d81d125be06)

**PPO Proximal Policy Optimization Rl Algorithm**

RL algorithm is called PPO used to perform the actual model weights updates based on the reward value assigned to a given prompt and completion. PPO updates the weights of the generateve model based on the reward value returned from the reward model. PPO will be used to optimize the RL policy against the reward model.

```
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model,                                                               
                                                               torch_dtype=torch.bfloat16,
                                                               is_trainable=True)

print(f'PPO model parameters to be updated (ValueHead + 769 params):\n{print_number_of_trainable_model_parameters(ppo_model)}\n')
print(ppo_model.v_head)
```

During PPO, only a few parameters will be updated. Specifically, the parameters of the ValueHead.


```
PPO model parameters to be updated (ValueHead + 769 params):

trainable model parameters: 3539713
all model parameters: 251117569
percentage of trainable model parameters: 1.41%

ValueHead(
  (dropout): Dropout(p=0.1, inplace=False)
  (summary): Linear(in_features=768, out_features=1, bias=True)
  (flatten): Flatten(start_dim=1, end_dim=-1)
)
```



**Prolem: To mitigate reward hacking**, which means that there exists a tendancy to ignore constraints and ack the rewards. Agent may learn to cheat and maximize the reward even if the chosen actions lead to an incorrect state.

A common techinique to avoid reward hacking is to first make a copy of the original instruct model before performing any reinforcement learninng or weight updates. You freeze the weights of copied model and use it as an immurable "reference model". During RLHF, every prompt is completed by both the frozen reference model and the model you are trying to fine-tune with RLHF. Next, two completions are compared to determine the statistical distance etween the two token-proaility distriutions. This distance is calculated using KL divergence. KL divergence quantifies how much the mutable, RLHF-tuned generative model is genneratinng completions thast diverge too far from the completions generated y immutable reference model. In short, if the fined-tunred model starts hacking the reward and generating of tokens that diverge too far from sequences that the reference model would generate, the fine-tuned model is penalized y the RL algorithm through a lower reward.

![image](https://github.com/user-attachments/assets/3eb81a02-f793-4020-b0e6-1988fc17bb7f)

**Using PEFT using RLHF**

PEFT can be used with RLHF to reduce the amount of compute and memory resourse required for the compute-intensive PPO algorithm. You would only need to update the model's much-
smaller PEFT adapter weights and not the full weights of the tunable model.

![image](https://github.com/user-attachments/assets/1057dca3-8d6b-47e5-b787-875c748f0db6)

![image](https://github.com/user-attachments/assets/84a1b0ea-4fe5-4536-8a69-c88f617d2ae5)



# Reference 
1. Coursera Courses https://www.coursera.org/learn/generative-ai-with-llms
2. Prompt-Tuning HuggingFace Doc https://huggingface.co/learn/cookbook/prompt_tuning_peft
3. BOOK: oreilly, Generative AI on AWS
4. https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb
5. 

