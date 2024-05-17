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
import multiprocessing
from multiprocessing import  Process, Manager, RLock

class Redis():
    def __init__(self):
        redis_config = {
            'host': '127.0.0.1',
            'port': 6379,
            'db': 0,
        }
        self.r = redis.StrictRedis(**redis_config)
        self.redis_clear()

    def redis_clear(self):
        if self.r.exists("response"):
            self.r.delete("response")
        if self.r.exists("request"):
            self.r.delete("request")
        if self.r.exists("id"):
            self.r.delete("id")

    def redis_hkeys(self,key):
        return self.r.hkeys(key)
    
    def redis_hget(self,key,field):
        return self.r.hget(key,field).decode("utf-8")
    
    def redis_hdel(self,key,field):
        self.r.hdel(key,field)

    def redis_hset(self,key,field,value):
        self.r.hset(key,field,value)



class MyProcess(Process,Redis):
    def __init__(self,ProcessID=None,lock=None):
        super(MyProcess,self).__init__()
        Redis.__init__(self,)
        self.lock = lock
        self.ProcessID = ProcessID

    def run(self):

        #####################加载Actor模型######################

        # 模型&tokenizer路径
        self.actor_model_path = "/nfs40/zh/model/production/chatglm2-6b/"

        # 加载tokenizer
        self.actor_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        # 生成模型
        self.actor_model_class = AutoModelForSeq2SeqLM
        self.actor_model = create_hf_model(self.actor_model_class,self.actor_model_path,self.actor_tokenizer, None)
        self.actor_model.to(torch.device("cuda",self.ProcessID))
        self.actor_model.eval()
        print("********************进程%d Actor模型加载完成********************" %self.ProcessID)
        """ 
        #####################加载Reward模型######################
        # 模型路径
        reward_path = "/nfs40/zh/model/production/reward-models/"
        # 模型名称
        reward_model_name = "chatglm-6b"
        # 模型路径
        self.reward_model_path = f"{reward_path}{reward_model_name}"
        # 加载tokenizer
        self.reward_tokenizer = load_hf_tokenizer(self.reward_model_path, fast_tokenizer=True)
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        # 加载模型
        self.reward_model = create_critic_model(self.reward_model_path,self.reward_tokenizer,None,0,True)
        self.reward_model.to(torch.device("cuda",self.ProcessID+1))
        self.reward_model.eval()
        print("********************进程%d Reward模型加载完成********************" %self.ProcessID)
        """
        while True:
            if Redis.redis_hkeys(self,"request"):
                self.lock.acquire()
                if not Redis.redis_hkeys(self,"request"):
                    self.lock.release()
                    continue
                request_keys = Redis.redis_hkeys(self,"request") # 列表
                uid = request_keys[0].decode("utf-8")
                prompt = Redis.redis_hget(self,"request",uid)
                Redis.redis_hdel(self,"request",uid)
                self.lock.release()
                # 以关键字列表是否为空判断请求队列是否有请求数据
                print("********************进程%d中模型处理端接受请求********************" %self.ProcessID)
                print("模型所在gpu：",next(self.actor_model.parameters()).device)
                print("请求uuid：",uid)
                print("请求数据：",prompt)
                start = time.time()
                outputs0,history = self.actor_model.chat(self.actor_tokenizer, prompt, history=[], max_length=512, do_sample=True,temperature=1.0)##temperature=0.5,repetition_penalty=2.0, temperature=0.12 ##num_beams=5,num_beam_groups=5,top_k=10,top_p=0.9,min_length=-1)
                outputs1,history = self.actor_model.chat(self.actor_tokenizer, prompt, history=[], max_length=512, do_sample=True,temperature=1.0)##temperature=0.5,repetition_penalty=2.0, temperature=0.12 ##num_beams=5,num_beam_groups=5,top_k=10,top_p=0.9,min_length=-1)
                print("模型处理时间：",time.time()-start)
                response = [outputs0,outputs1]
                """     
                # 生成分数
                for i in range(len(response)):
                    token = preprocess_function_reward(self.reward_tokenizer,prompt,response[i])
                    token = to_device(token,torch.device("cuda",self.ProcessID+1))
                    score = self.reward_model.forward_value(**token,eval_reward=True)
                    score = score["chosen_end_scores"]
                    response[i] = response[i]+"-->"+str("%.2f"%score)
                """
                # 处理完成后将响应数据添加进响应队列,以&作为分割符链接所有响应
                Redis.redis_hset(self,"response",uid,"&".join(response))
                # 处理完成后将uuid对应的数据在请求队列中删除
                print("********************进程%d请求处理结束********************" %self.ProcessID)
                



if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    processs = []
    totalProcess = 2
    lock = RLock()
    for i in range(totalProcess):
        print("********************设置进程：%d ********************" %i)
        process = MyProcess(i,lock)
        processs.append(process)
        print("********************进程：%d 设置完成********************" %i)
    for i in range(totalProcess):
        processs[i].start()
        print("********************开启进程：%d ********************" %i)
