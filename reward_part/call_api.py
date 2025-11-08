from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import json

app = FastAPI()

class EvaluateRequest(BaseModel):
    think_content: str
    extracted_answer: str
    ground_truth: str
    question: str

# 加载模型
print("正在加载Qwen模型...")
model_path = "your model path"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("模型加载完成!")

def create_evaluation_prompt(think_content, question, extracted_answer, ground_truth):
    """创建评估提示词"""
    
    prompt = f"""请作为专业评估专家，对思维链的质量给分：

问题：{question}

思维链内容：
{think_content}

模型给出的答案：{extracted_answer}
标准答案：{ground_truth}

请从以下5个维度给分（每个维度最低0分，最高也要低于0.2分）：
1. 逻辑连贯性：推理步骤是否逻辑清晰
2. 步骤完整性：是否覆盖所有关键步骤  
3. 数学准确性：计算过程是否连贯
4. 问题相关性：是否围绕问题展开
5. 表达清晰度：表达是否清晰简洁

请给出每个维度的分数，然后计算总分。

请严格按照以下JSON格式返回，不需要给任何解析：
{{
    "scores": {{
        "logic": {{"score": 分数}},
        "completeness": {{"score": 分数}},
        "math_accuracy": {{"score": 分数}},
        "relevance": {{"score": 分数}},
        "clarity": {{"score": 分数}}
    }},
    "think_score": 总分
}}"""

    return prompt

def call_llm_judge(prompt):
    """调用本地LLM进行评分"""
    messages = [
        {"role": "system", "content": "你是一个专业数学问题的评估专家，只给出分数，不给任何解析。"},
        {"role": "user", "content": prompt}
    ]
    
    # 构建输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("LLM Judge Response:", response)
    
    # 提取JSON
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # 如果解析失败，返回默认值
    return {
        "scores": {
            "logic": {"score": 5, "reason": "解析失败"},
            "completeness": {"score": 5, "reason": "解析失败"},
            "math_accuracy": {"score": 5, "reason": "解析失败"},
            "relevance": {"score": 5, "reason": "解析失败"},
            "clarity": {"score": 5, "reason": "解析失败"}
        },
        "think_score": 5
    }

def evaluate_accuracy(extracted, ground_truth):
    """评估答案准确性"""
    if not extracted or not ground_truth:
        return 0
    
    # 直接比较
    if extracted.strip() == ground_truth.strip():
        return 1
    
    # 尝试数值比较
    try:
        ext_clean = re.sub(r'[^\d.]', '', extracted)
        gt_clean = re.sub(r'[^\d.]', '', ground_truth)
        if ext_clean and gt_clean and float(ext_clean) == float(gt_clean):
            return 1
    except:
        pass
    
    return 0

@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    # 评估思维链
    prompt = create_evaluation_prompt(
        request.think_content,
        request.question, 
        request.extracted_answer,
        request.ground_truth
    )
    
    think_result = call_llm_judge(prompt)
    think_score = think_result.get("think_score", 0)
    
    # 评估准确性
    accuracy_score = evaluate_accuracy(request.extracted_answer, request.ground_truth)
    
    # 计算综合分数
    final_score = 0.2 * think_score + 0.8 * accuracy_score
    
    return {
        "think": think_score,
        "accuracy": accuracy_score,
        "score": final_score,
        "think_details": think_result
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)