import os
import string
import sys
sys.path.append("~/rlhf/applications/DeepSpeed-Chat/training")
from fastapi import FastAPI, Request
import uvicorn, json, datetime, asyncio
import torch
import time
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
# os.path.append() : 添加自己的引用模块搜索目录（路径）到系统的环境变量
# os.path.pardir : 获取当前目录的父目录（上一级目录）
# os.path.dirname(__file__) : 当前运行脚本所在位置（路径）
# os.path.abspath : 源码解释-取决于os.getcwd,如果是一个绝对路径，就返回，如果不>是绝对>路径，根据编码执行getcwd/getcwdu.然后把path和当前工作路径连接起来．
# os.getcwd : 获取当前运行脚本所在目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from training.utils.model.model_utils import create_hf_model,create_critic_model
from training.utils.data.data_utils_chatglm2 import preprocess_function_reward
from training.utils.utils import load_hf_tokenizer,to_device
import redis
from concurrent.futures import ThreadPoolExecutor, wait

from typing import Dict
import numpy as np
import ray


def load_actor_model():
    # 模型路径
    path = "/data9/NFS/zh/model/training/actor-models/"
    # 模型名称
    step1_model_name = "chatglm2-6b"
    # 模型&tokenizer路径
    actor_model_path = f"{path}{step1_model_name}"
    # 加载tokenizer
    actor_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    # 生成模型
    actor_model_class = AutoModelForSeq2SeqLM
    actor_model = create_hf_model(actor_model_class,actor_model_path,actor_tokenizer, None)
    # actor_model.to(device)
    actor_model.eval()
    print("********************Actor模型加载完成********************")
    return actor_tokenizer, actor_model


def load_reward_model():
    # 模型路径
    path = "/data9/NFS/zh/model/training/reward-models"
    # 模型名称
    reward_model_name = "chatglm2-6b"
    # 模型路径
    reward_model_path = f"{path}{reward_model_name}"
    # 加载tokenizer
    reward_tokenizer = load_hf_tokenizer(reward_model_path)
    # reward_tokenizer.pad_token = reward_tokenizer.eos_token
    # 加载模型
    reward_model = create_critic_model(reward_model_path,reward_tokenizer,None,0,True)
    # reward_model.to(device)
    reward_model.eval()
    print("********************Reward模型加载完成********************")
    return reward_tokenizer,reward_model



class ModelPredictor:
    def __init__(self):
        # from transformers import pipeline
        # Set "cuda:0" as the device so the Huggingface pipeline uses GPU.
        self.actor_tokenizer, self.actor_model = load_actor_model()

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        print("batch:",batch)
        predictions = self.actor_model.chat(self.actor_tokenizer,list(batch["data"]), max_length=20, num_return_sequences=1)
        batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
        # 处理完成后将响应数据添加进响应队列,以&作为分割符链接所有响应
        r.hset("response",uid,"&".join(response))
        # 处理完成后将uuid对应的数据在请求队列中删除
        r.hdel("request",uid)
        print("********************请求处理结束********************")
    

        return batch


def get_response(request_dict=None):
    # 以关键字列表是否为空判断请求队列是否有请求数据
    for uid,prompt in request_dict.items():
        print("********************模型处理端接受请求********************")
        print("请求uuid：",uid)
        print("请求数据：",prompt)

        outputs0,history = actor_model.chat(actor_tokenizer, prompt, history=[], max_length=512, do_sample=True,temperature=1.0)##temperature=0.5,repetition_penalty=2.0, temperature=0.12 ##num_beams=5,num_beam_groups=5,top_k=10,top_p=0.9,min_length=-1)
        outputs1,history = actor_model.chat(actor_tokenizer, prompt, history=[], max_length=512, do_sample=True,temperature=1.0)##temperature=0.5,repetition_penalty=2.0, temperature=0.12 ##num_beams=5,num_beam_groups=5,top_k=10,top_p=0.9,min_length=-1)
        
        response = [outputs0,outputs1]
        for i in range(len(response)):
            token = preprocess_function_reward(reward_tokenizer,prompt,response[i])
            score = reward_model.forward_value(token,eval_reward=True)
            response[i] = response[i]+str("%.2f"%score)
        
        # 处理完成后将响应数据添加进响应队列,以&作为分割符链接所有响应
        r.hset("response",uid,"&".join(response))
        # 处理完成后将uuid对应的数据在请求队列中删除
        r.hdel("request",uid)
        print("********************请求处理结束********************")
    

def main():
    # worker_count = 50
    # with ThreadPoolExecutor(max_workers=worker_count) as pool:
    while True:
        request_list = r.hgetall("request")
        print(f"request_list {request_list}")
        ds = ray.data.from_dict(request_list)
        print(f"ds {ds}")
        if request_list:
            predictions = ds.map_batches(
                            ModelPredictor,
                            num_gpus=1,
                            # Specify the batch size for inference.
                            # Increase this for larger datasets.
                            batch_size=1,
                            # Set the ActorPool size to the number of GPUs in your cluster.
                            compute=ray.data.ActorPoolStrategy(size=84),
            )
            print(predictions)

            # for uuid, prompt in request_list.items():
            #     request_dict = {uuid.decode("utf-8"): prompt.decode("utf-8")}
            #     r.hdel("request",uuid.decode("utf-8"))
            #     pool.submit(get_response, request_dict)
        else:
            time.sleep(0.1)
    

if __name__ == "__main__":
    redis_config = {
    'host': '127.0.0.1',
    'port': 6379,
    'db': 0,
     }
    r = redis.StrictRedis(**redis_config)
    # device = torch.device("cuda:0")
    actor_tokenizer, actor_model = load_actor_model()
    # reward_tokenizer,reward_model = load_reward_model()
    if r.exists("response"):
        r.delete("response")
    if r.exists("request"):
        r.delete("request")
    if r.exists("id"):
        r.delete("id")

    main()
