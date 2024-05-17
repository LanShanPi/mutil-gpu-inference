import os
import string
import sys
sys.path.append(r"/home/duser/hzl/rlhf/applications/DeepSpeed-Chat/training")
from fastapi import FastAPI, Request
import uvicorn, json, datetime, asyncio
import torch
import time
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_hf_model,create_critic_model
from utils.data.data_utils_chatglm import preprocess_function_reward
from utils.utils import load_hf_tokenizer,to_device
import redis
from concurrent.futures import ThreadPoolExecutor, wait
import copy
def load_actor_model():
    # 模型路径
    path = "/nfs40/zh/model/production/"
    # 模型名称
    step1_model_name = 'chatglm-6b'

    # 模型&tokenizer路径
    actor_model_path = f'{path}{step1_model_name}'

    # 加载tokenizer
    actor_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # 生成模型
    actor_model_class = AutoModelForSeq2SeqLM
    actor_model = create_hf_model(actor_model_class,actor_model_path,actor_tokenizer, None)
    actor_model_list = []

    for i in range(4,8):
        model = copy.deepcopy(actor_model.to(torch.device("cuda",i)))
        model.eval()
        actor_model_list.append(model)

    print("********************Actor模型加载完成********************")
    return actor_tokenizer, actor_model_list

def load_reward_model():
    # 模型路径
    path = "/nfs40/zh/model/production/reward-models/"
    # 模型名称
    reward_model_name = "chatglm-6b"
    # 模型路径
    reward_model_path = f"{path}{reward_model_name}"
    # 加载tokenizer
    reward_tokenizer = load_hf_tokenizer(reward_model_path, fast_tokenizer=True)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    # 加载模型
    reward_model = create_critic_model(reward_model_path,reward_tokenizer,None,0,True)
    reward_model_list = []
    for i in range(4,8):
        model = copy.deepcopy(reward_model.to(torch.device("cuda",i)))
        model.eval()
        reward_model_list.append(model)
    print("********************Reward模型加载完成********************")
    return reward_tokenizer,reward_model_list

def get_response(request_dict=None,actor_model=None,reward_model=None,ind=None):
    # 以关键字列表是否为空判断请求队列是否有请求数据
    for uid,prompt in request_dict.items():
        print("********************模型处理端接受请求********************")
        print("请求uuid：",uid)
        print("请求数据：",prompt)
        outputs0,history = actor_model.chat(actor_tokenizer, prompt, history=[], max_length=200, do_sample=True,temperature=1.0)##temperature=0.5,repetition_penalty=2.0, temperature=0.12 ##num_beams=5,num_beam_groups=5,top_k=10,top_p=0.9,min_length=-1)
        outputs1,history = actor_model.chat(actor_tokenizer, prompt, history=[], max_length=200, do_sample=True,temperature=1.0)##temperature=0.5,repetition_penalty=2.0, temperature=0.12 ##num_beams=5,num_beam_groups=5,top_k=10,top_p=0.9,min_length=-1)
        print("答案已生成")
        response = [outputs0,outputs1]
        """
        for i in range(len(response)):
            token = preprocess_function_reward(reward_tokenizer,prompt,response[i])
            print("token已生成")
            device = torch.device("cuda",ind)
            token = to_device(token,device)
            print("token已加载到gpu")
            score = reward_model.forward_value(**token,eval_reward=True)
            print("分数已生成")
            score = score["chosen_end_scores"]
            response[i] = response[i]+"-->"+str("%.2f"%score)
        print("奖励已生成")
        """
        # 处理完成后将响应数据添加进响应队列,以&作为分割符链接所有响应
        r.hset("response",uid,"&".join(response))
        # 处理完成后将uuid对应的数据在请求队列中删除
        r.hdel("request",uid)
        print("********************请求处理结束********************")

def main():
    print("********************准备就绪可以开始测试********************")
    worker_count = 50
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        while True:
            request_list = r.hgetall("request")
            if request_list:
                num = 0
                for uuid, prompt in request_list.items():
                    request_dict = {uuid.decode("utf-8"): prompt.decode("utf-8")}
                    r.hdel("request",uuid.decode("utf-8"))
                    ind = num%4
                    actor_model = actor_model_list[ind]
                    # reward_model = reward_model_list[ind]
                    pool.submit(get_response, request_dict,actor_model)
                    num += 1
            else:
                time.sleep(0.1)


if __name__ == "__main__":
    redis_config = {
        'host': '127.0.0.1',
        'port': 6379,
        'db': 1,
     }
    r = redis.StrictRedis(**redis_config)
    actor_tokenizer,actor_model_list = load_actor_model()
    # reward_tokenizer,reward_model_list = load_reward_model()
    if r.exists("response"):
        r.delete("response")
    if r.exists("request"):
        r.delete("request")
    if r.exists("id"):
        r.delete("id")
    print("********************数据库残存数据清除完成********************")
    main()
