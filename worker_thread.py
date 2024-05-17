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
import threading
import multiprocessing
from multiprocessing import  Process ,Lock


class MyThread(threading.Thread):
    def __init__(self,threadID=None,lock=None):
        threading.Thread.__init__(self)
        self.lock = lock
        self.threadID = threadID
        #####################加载Reward模型######################
        # 模型路径
        actor_path = "/nfs40/zh/model/production/chatglm-6b/"
        # 模型名称
        step1_model_name = "actor"

        # 模型&tokenizer路径
        actor_model_path = f"{actor_path}{step1_model_name}"

        # 加载tokenizer
        self.actor_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        # 生成模型
        actor_model_class = AutoModelForSeq2SeqLM
        self.actor_model = create_hf_model(actor_model_class,actor_model_path,self.actor_tokenizer, None)
        self.actor_model.to(torch.device("cuda",self.threadID+2))
        self.actor_model.eval()
        print("********************线程%d Actor模型加载完成********************" %self.threadID)
        
        """
        #####################加载Reward模型######################
        # 模型路径
        reward_path = "/nfs40/zh/model/production/reward-models/"
        # 模型名称
        reward_model_name = "chatglm-6b"
        # 模型路径
        reward_model_path = f"{reward_path}{reward_model_name}"
        # 加载tokenizer
        self.reward_tokenizer = load_hf_tokenizer(reward_model_path, fast_tokenizer=True)
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        # 加载模型
        self.reward_model = create_critic_model(reward_model_path,self.reward_tokenizer,None,0,True)
        self.reward_model.to(torch.device("cuda",self.ThreadID))
        self.reward_model.eval()
        print("********************线程%d Reward模型加载完成********************" %self.ThreadID)
        """

    def run(self):
        while True:
            # print("********************线程%d循环等待请求数据********************" %self.threadID)
            if r.hkeys("request"):
                self.lock.acquire()
                if not r.hkeys("request"):
                    self.lock.release()
                    continue
                request_keys = r.hkeys("request") # 列表
                uid = request_keys[0].decode("utf-8")
                prompt = r.hget("request",uid).decode("utf-8")
                r.hdel("request",uid)
                self.lock.release()
                # 以关键字列表是否为空判断请求队列是否有请求数据
                # print("********************线程%d中模型处理端接受请求********************" %self.threadID)
                # print("模型所在gpu：",next(self.actor_model.parameters()).device)
                # print("请求uuid：",uid)
                # print("请求数据：",prompt)
                start = time.time()
                outputs0,history = self.actor_model.chat(self.actor_tokenizer, prompt, history=[], max_length=200, do_sample=True,temperature=1.0)##temperature=0.5,repetition_penalty=2.0, temperature=0.12 ##num_beams=5,num_beam_groups=5,top_k=10,top_p=0.9,min_length=-1)
                outputs1,history = self.actor_model.chat(self.actor_tokenizer, prompt, history=[], max_length=200, do_sample=True,temperature=1.0)##temperature=0.5,repetition_penalty=2.0, temperature=0.12 ##num_beams=5,num_beam_groups=5,top_k=10,top_p=0.9,min_length=-1)
                # print("模型处理时间：",time.time()-start)
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
                # print("********************线程%d请求处理结束********************" %self.threadID)
            



if __name__ == "__main__":
    redis_config = {
        'host': '127.0.0.1',
        'port': 6379,
        'db': 1,
     }
    r = redis.StrictRedis(**redis_config)
    
    if r.exists("response"):
        r.delete("response")
    if r.exists("request"):
        r.delete("request")
    if r.exists("id"):
        r.delete("id")
    print("********************数据库残存数据清除完成********************")
    threads = []
    totalThread = 2
    lock = threading.Lock()
    for i in range(totalThread):
        print("********************设置线程：%d ********************" %i)
        thread = MyThread(i,lock)
        threads.append(thread)
        print("********************线程：%d 设置完成********************" %i)
    for i in range(totalThread):
        threads[i].start()
        print("********************开启线程：%d ********************" %i)
    print("********************准备工作就绪可以开始测试********************")
