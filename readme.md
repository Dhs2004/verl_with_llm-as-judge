# Custom Reward Extension for [verl](https://github.com/volcengine/verl)

## 简介

官方版本的 **verl** 框架中不支持通过 API 对模型的 response 进行打分，也不支持自定义的 LLM 打分机制。  
为了使 **rollout 过程中的 reward 更加模块化与可扩展**，本仓库对 verl 进行了拓展：

-  将 reward 模块独立出来，提供统一的 `reward_api` 接口。  
-  支持 **LLM-as-Judge**，可自定义提示词模板（prompt template）。  
-  更易于集成 rule-based、LLM-based 或混合打分逻辑。  

---

##  使用方法

1. **参考 verl 官方文档** 配置基础环境  
    [Verl 官方环境配置说明](https://verl.readthedocs.io/en/latest/getting_started/installation.html)

2. 进入本模块目录：

   ```bash
   cd reward_part
   ```

---

# Reward API 使用说明与训练流程

##  第三步：启动 Reward API（若使用 LLM-as-Judge 模式）

若你希望使用 **LLM-as-Judge** 来对模型生成的 response 进行自动化评估，请在终端进入 `reward_part` 目录后执行以下命令：

```bash
python call_api.py
```

此脚本会启动一个本地的 HTTP 服务，用于对外暴露 reward 打分接口。  
默认启动地址为：

```
http://0.0.0.0:6009/get_reward2
```

运行成功后，你会在终端看到类似的输出信息：

```
Reward API running on port 6009...
Use Ctrl+C to stop the server.
```

这表示你调用大模型作为评分者服务已成功启动。此时，训练脚本中的 `reward_part.reward_api` 参数即可指向该地址，从而在强化学习过程中动态获取评分。

>  提示：  
>
> - 若你计划使用基于规则的打分逻辑（rule-based reward），请跳过此步并直接执行下一步。  
>
> - 如果需要修改端口号，可以在 `call_api.py` 中更改默认配置，例如：
>
>   ```python
>   app.run(host="0.0.0.0", port=6010)
>   ```

---

##  第四步：启动自定义规则打分服务

若你希望采用自定义的基于规则的打分逻辑（例如关键字匹配、正则判断、数值范围评分等），请修改`rw_custom.py`执行：

```bash
python rw_custom.py
```

此脚本将启动一个 rule-based reward 评分服务，同样会输出实际的接口地址，例如：

```
Custom Rule-Based Reward API listening on: http://0.0.0.0:6010/get_reward
```

---

##  第五步：修改训练脚本端口地址

复制第 4 步终端中输出的接口地址，替换掉下面会给出的脚本的`reward_api`部分：

```bash
reward_model.reward_api=http://0.0.0.0:6009/get_reward2
```

将 `6009` 改为实际的端口号。

---

##  第六步：执行训练命令

在修改完成后，运行以下命令以启动 GRPO 训练：

```bash
bash ./train_grpo_cr.sh
```

这将自动加载自定义的 reward API 服务，对模型输出的 response 进行打分与优化。

---

##  训练脚本示例

```bash
# train grpo use custom reward api
# base param doc: https://verl.readthedocs.io/en/latest/examples/config.html#config-explain-page
# add param: reward_model.reward_api=http://10.xxx.0.xxx:6009/get_reward2 
export CUDA_VISIBLE_DEVICES="0,1,2,3,5,6,7"

set -x
ray stop
export TMPDIR=your tmpdir
mkdir -p $TMPDIR

# 设置 Ray 使用新临时目录
export RAY_TMPDIR=$TMPDIR

ray start \
  --head \
  --node-ip-address=0.0.0.0 \
  --port=6378 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --ray-debugger-external \
  --num-gpus 4

model_path=your_model_path  
cur_task=verl_custom_test  
save_model_checkpoint=your_model_path/$cur_task  

echo "在主节点上启动训练..."

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=remax \
    data.train_files=train_path \
    data.val_files=val_path \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.truncation=right \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_api=http://0.0.0.0:6009/get_reward2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_custom' \
    trainer.experiment_name=$cur_task \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir=$save_model_checkpoint \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@

```

---

##  功能总结

| 模块                       | 功能                             | 可自定义          |
| -------------------------- | -------------------------------- | ----------------- |
| `reward_part/call_api.py`  | 启动 LLM-as-Judge 的 reward 服务 |  Prompt 模板     |
| `reward_part/rw_custom.py` | 自定义 reward 实现（`rm-based reward or rule-based reward`）          |  规则逻辑        |
| `verl_custom_train.sh`     | 调用自定义 reward 的训练入口     |  Reward API 地址 |

---

##  延伸阅读

- [verl 官方仓库](https://github.com/volcengine/verl)  
- [Verl 文档中心](https://verl.readthedocs.io/en/latest/)