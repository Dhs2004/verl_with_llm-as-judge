# 服务器端代码 (llm_as_judge.py)
from fastapi import Depends, FastAPI, HTTPException, Request
import re
import asyncio
from pydantic import BaseModel
import requests
import json
from converter import process_evaluation_data
app = FastAPI()

@app.post("/get_reward2")
async def get_reward2(request: Request):
    """使用自定义的reward模型，提供反馈接口"""
    print("=== REWARD API CALLED ===")
    json_data = await request.json()
    json_data = process_evaluation_data(json_data)
    def wrap_process_data():
        url = "http://localhost:8001/evaluate"
        response = requests.post(url, json=json_data)
        result=response.json()
        think_score = result.get("think",0)
        accuracy_score= result.get("accuracy",0)
        score = result.get("score",0)
        think_detail = result.get("think_details","")

            
        score = {
            "think_socre": float(think_score),
            "accuracy": float(accuracy_score),
            "score": float(score),
            "think_details": think_detail
        }
        
        # print(f"📊 最终分数: {score}")
        return score
    result = await asyncio.to_thread(wrap_process_data)
    return result
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6009)
