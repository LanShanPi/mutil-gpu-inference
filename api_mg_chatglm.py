import datetime
import json
from redis import asyncio as aioredis
import uuid
from typing import Optional
import os
import threading
import uuid
import redis
import string
import sys
from fastapi import FastAPI, Request
import uvicorn, json, datetime, asyncio
import torch
import time

app = FastAPI(title="RLHF",description="A api for inference use RLHF Model",version="2023-06-10")


@app.post("/single")
async def single_create_item(request: Request):
    # 处理提示信息
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    messages =json_post_list.get('messages')
    message = messages[-1]
    prompt = message['content']
    if "prefix" in message:
        prompt = message["prefix"] + message['content']

    # 生成回复
    uid = str(uuid.uuid1())
    local_single_uuid_list.append(uid)
    await r.hset("request",uid,prompt)
    print("********************收到单问题请求数据********************")
    print("请求数据uuid：",uid)
    print("请求数据：",prompt)
    start = time.time()
    while True:
        response_queue = await r.hgetall("response")
        if not response_queue:
            time.sleep(0.1)
            continue
        for uid in local_single_uuid_list:
            response = await r.hget("response", uid)
            if response == None:
                time.sleep(0.1)
                continue
            # 从响应队列中根据uuid取出响应数据
            response = response.decode("utf-8").strip("[").strip("]").split("&")
            # 从响应队列中根据uuid删除已取出的数据
            await r.hdel("response",uid)
            # 从局部uuid列表中删除uid
            local_single_uuid_list.remove(uid)

            answer = {
                "response": response,
                "version":version
                }
            print("********************单问题模型端响应数据如下********************")
            print("请求uuid：", uid)
            print("请求数据：",prompt)
            print("响应：",response)
            print("响应时间：",time.time()-start)
            return answer


@app.post("/multi")
async def multi_create_item(request: Request):
    # 处理提示信息
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    messages = json_post_list.get('messages')
    # 用于存储当前批量请求的uuid
    local_multi_uuid_list = []
    #用于存储当前批量请求的提示
    prompt = []
    # 直接将数据放入redis，同一个条数据中的id和提示对应同一个uuid
    for i in range(len(messages)):
        uid = str(uuid.uuid1())
        await r.hset("id",uid,messages[i]["id"])
        if "prefix" in messages[i]:
            await r.hset("request", uid, messages[i]["prefix"]+messages[i]["content"])
            prompt.append(messages[i]["prefix"]+messages[i]["content"])
        else:
            await r.hset("request", uid, messages[i]["content"])
            prompt.append(messages[i]["content"])
        local_multi_uuid_list.append(uid)
    prompts.append(prompt[::])
    total_local_multi_uuid_list.append(local_multi_uuid_list[::])
    print("********************收到批量请求数据********************")
    print("请求数据uuid：",total_local_multi_uuid_list[-1])
    print("请求数据：",prompts[-1])
    # 存储返回数据
    response_list = []
    total_response_list.append(response_list)
    start = time.time()
    while True:
        response_queue = await r.hgetall("response")
        if not response_queue:
            time.sleep(0.1)
            continue
        for i in range(len(total_local_multi_uuid_list)):
            for uid in total_local_multi_uuid_list[i]:
                # 如果没有对应的响应信息则跳过
                response = await r.hget("response", uid)
                if response == None:
                    time.sleep(0.1)
                    continue
                # 处理id
                ids = await r.hget("id", uid)
                ids = ids.decode("utf-8")
                # 处理响应信息
                response = response.decode("utf-8").strip("[").strip("]").split("&")
                # 组成返回数据
                total_response_list[i].append(
                        {"id": ids, "response": [response[0], response[1]], "version": version})

                # 从本地uuid列表中删除已经取出的响应数据
                total_local_multi_uuid_list[i].remove(uid)
                # 从响应队列中删除uid对应的数据
                await r.hdel("response", uid)
                # 从id队列中删除uid对应的id
                await r.hdel("id", uid)
            # 当本次批量请求的所有数据均处理完毕后再返回请求
            if len(total_local_multi_uuid_list[i]) == 0:
                answer = {
                    "response": total_response_list[i],
                }
                print("********************多问题模型端响应数据如下********************")
                print("当前批次请求数据：", prompts[i])
                print("当前批次请求响应：", total_response_list[i])
                print("当前批次请求响应时间：",time.time()-start)
                del total_response_list[i]
                del prompts[i]
                del total_local_multi_uuid_list[i]
                return answer

if __name__ == "__main__":
    # 获取模型版本号
    with open("/nfs40/zh/model/production/chatglm2-6b/config.json", 'r') as f:
        version = json.load(f)["version"]
    # 针对不同的处理模块创建队列
    redis_config = {
        'host': '127.0.0.1',
        'port': 6379,
        'db': 0,
    }
    r = aioredis.StrictRedis(**redis_config)
    # 用于存储single的uuid
    local_single_uuid_list = []
    # 用于存储所有批量处理中的提示
    prompts = []
    # 本地存储所有的批量uuid,形式为[local_multi_uuid_list1,local_multi_uuid_list2...]
    total_local_multi_uuid_list = []
    # 存储所有的响应列表，形式为[response_list1,response_list2,...]
    total_response_list = []
    # 清除数据库数据
    if r.exists("response"):
        r.delete("response")
    if r.exists("request"):
        r.delete("request")
    if r.exists("id"):
        r.delete("id")
    print("********************数据库残存数据清除完成********************")
    # 8536~8539，对应的对外映射是118.184.171.68:38686~38689
    uvicorn.run(app, host="0.0.0.0", port=8536)
