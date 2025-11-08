# converter.py
import re

def convert_response_format(response_str):
    """
    将response_str转换为<think>和<answer>格式
    """
    # 找到最后一个####的位置
    last_hash_pos = response_str.rfind('####')
    
    if last_hash_pos != -1:
        # 分割思考部分和答案部分
        think_part = response_str[:last_hash_pos].strip()
        
        # 提取答案部分
        answer_section = response_str[last_hash_pos + 4:]  # +4 跳过 '####'
        
        # 移除<|im_end|>标记
        if '<|im_end|>' in answer_section:
            answer_part = answer_section.split('<|im_end|>')[0].strip()
        else:
            answer_part = answer_section.strip()
    else:
        # 没有找到####，整个内容作为思考部分
        think_part = response_str.strip()
        answer_part = ""
    
    # 清理答案中的特殊字符
    answer_part = re.sub(r'[\$\\]', '', answer_part)
    
    # 构建目标格式
    converted_format = f"""<think>
{think_part}
</think>

<answer>\\boxed{{{answer_part}}}</answer>"""
    
    return think_part, answer_part
def process_evaluation_data(evaluation_data):
    """
    处理完整的评估数据格式
    """
    # 提取关键信息
    response_str = evaluation_data['response_str']
    ground_truth = evaluation_data['ground_truth']
    question = evaluation_data['extra_info']['question']
    
    # 转换格式
    think_content, extracted_answer = convert_response_format(response_str)
    
    # 准备发送到评分服务器的数据
    judge_data = {
        "think_content": think_content,
        "extracted_answer": extracted_answer,
        "ground_truth": ground_truth,
        "question": question
    }
    
    return judge_data