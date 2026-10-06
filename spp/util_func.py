import re
import datetime
import os
import argparse
import logging
import traceback
import json
import jsonlines
import csv
import logging
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import google.generativeai as genai
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
import os
import openai
import requests
import time

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

if os.environ["task"] == 'writing':
    from agent_prompt_writing import *
elif os.environ["task"] == 'logic':
    from agent_prompt_logic import *


def message_construction(model_name, prompt):
    if model_name != 'gemini':
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "user", "parts": [prompt]},
        ]
    return messages


def refine_func(ind, question, answer, model_name, data_type):
    i = 0
    answer_list = []
    while i < ind:
        prompt = feedback_agent_prompt.format(question=question, answer=answer)
        messages = message_construction(model_name, prompt)
        feedback_description = evaluator_construction(messages, model_name, question, data_type)

        prompt = self_refine_agent_prompt.format(question=question, answer=answer, feedback=feedback_description)
        messages = message_construction(model_name, prompt)
        new_answer = evaluator_construction(messages, model_name, question, data_type)

        answer_list.append({
            "time": i,
            "results": answer,
            "refine": feedback_description,
            "new_results": new_answer
        })
        # pdb.set_trace()
        answer = new_answer
        i = i + 1

    return answer_list, answer


def spp_func(ind, question, answer, model_name, data_type):
    prompt = persona_gen_agent_prompt.format(question=question)
    messages = message_construction(model_name, prompt)
    persona_ls = evaluator_construction(messages, model_name, question, data_type)
    persona_ls = persona_ls.split(";")
    # pdb.set_trace()
    suggest_inital = []
    for persona in persona_ls:
        prompt = suggest_gen_agent_prompt.format(question=question, persona=persona)
        messages = message_construction(model_name, prompt)
        suggest = evaluator_construction(messages, model_name, question, data_type)
        suggest_inital.append(f"{persona.strip()}: {suggest.strip()}")

    prompt = spp_agent_prompt.format(question=question, suggestion='\n'.join(suggest_inital))
    messages = message_construction(model_name, prompt)
    answer = evaluator_construction(messages, model_name, question, data_type)
    answer_list = []
    answer_list.append(
        {
            "persona": persona_ls,
            "suggest_inital": suggest_inital,
            "answer": answer,
        }
    )
    i = 0
    while i < ind:
        suggest_temp = []
        for persona in persona_ls:
            prompt = spp_feedback_agent_prompt.format(question=question, answer=answer, persona=persona)
            messages = message_construction(model_name, prompt)
            spp_feedback_description = evaluator_construction(messages, model_name, question, data_type)
            if spp_feedback_description.strip() == "Well Done!":
                continue
            else:
                suggest_temp.append(f"{persona.strip()}: {spp_feedback_description.strip()}")
        if len(suggest_temp) == 0:
            # pdb.set_trace()
            break
        prompt = spp_self_refine_agent_prompt.format(question=question, answer=answer,
                                                     suggestion='\n'.join(suggest_temp))
        messages = message_construction(model_name, prompt)
        new_answer = evaluator_construction(messages, model_name, question, data_type)

        answer_list.append({
            "time": i,
            "answer": answer,
            "suggest": suggest_temp,
            "new_answer": new_answer
        })
        # pdb.set_trace()
        answer = new_answer
        i = i + 1

    return answer_list, answer


def collaboration_func(ind, question, answer, model_name, data_type):
    i = 0
    answer_list = []
    description_ls = []
    while i < ind:
        flag = 0
        while True:
            prompt = meta_agent_prompt.format(question=question, answer=answer, description='\n-'.join(description_ls))
            messages = message_construction(model_name, prompt)
            description = evaluator_construction(messages, model_name, question, data_type)

            prompt = check_agent_prompt.format(question=question, description_ls='\n-'.join(description_ls), description=description)
            messages = message_construction(model_name, prompt)
            check_result = evaluator_construction(messages, model_name, question, data_type)
            print(check_result)
            if 'discard' not in check_result.lower() or flag > 3:
                description_ls.append(description)
                break
            flag += 1


        prompt = multi_agent_prompt.format(question=question, description=description)
        messages = message_construction(model_name, prompt)
        sub_answer = evaluator_construction(messages, model_name, question, data_type)

        prompt = refine_agent_prompt.format(question=question, description=description,
                                            old_answer=answer, new_answer=sub_answer)
        messages = message_construction(model_name, prompt)
        new_answer = evaluator_construction(messages, model_name, question, data_type)

        answer_list.append({
            "time": i,
            "results": answer,
            "description": description,
            "sub_answer": sub_answer,
            "new_results": new_answer
        })

        answer = new_answer
        i = i + 1

    return answer_list, answer


def evaluator_construction(messages, model_name, prompt, data_type='azure'):
    ind = 0
    while True:
        try:
            if data_type == "azure":
                openai.api_key = os.environ["OPENAI_API_KEY"]
                openai.api_base = os.environ["OPENAI_API_BASE"]
                openai.api_type = "azure"
                openai.api_version = os.environ["OPENAI_API_VERSION"]
                result = openai.ChatCompletion.create(
                    engine=model_name,
                    messages=messages,
                    temperature=0,
                    stop=None)
                clean_result = result["choices"][0]["message"]["content"]
            elif data_type == 'gemini':
                generation_config = {
                    "temperature": 0,
                }
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                # pdb.set_trace()
                model = genai.GenerativeModel('gemini-pro', generation_config=generation_config,
                                              safety_settings=safety_settings)
                result = model.generate_content(messages)
                clean_result = result.text
            elif data_type == 'openai':
                openai.api_key = os.environ["OPENAI_API_KEY"]
                result = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    stop=None)
                clean_result = result["choices"][0]["message"]["content"]
            elif data_type == 'small':
                # Local model via Ollama's native API (http://localhost:11434/api/chat).
                # Native API is used instead of the OpenAI-compat endpoint because the
                # compat layer silently ignores "options" (num_ctx/num_gpu) and "think"
                # is unreliable there; both are confirmed to work via /api/chat.
                ollama_base = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
                think = os.environ.get("OLLAMA_THINK", "false").strip().lower() == "true"
                num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
                num_gpu = int(os.environ.get("OLLAMA_NUM_GPU", "999"))
                max_tokens = int(os.environ.get("OLLAMA_MAX_TOKENS", "512"))
                temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0"))

                options = {
                    "num_ctx": num_ctx,
                    "num_gpu": num_gpu,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
                # Optional sampling-param passthrough: only set if the caller asked
                # for a specific value, so default-config runs' options dict is
                # unchanged from what earlier logs already recorded.
                if "OLLAMA_REPEAT_PENALTY" in os.environ:
                    options["repeat_penalty"] = float(os.environ["OLLAMA_REPEAT_PENALTY"])
                if "OLLAMA_PRESENCE_PENALTY" in os.environ:
                    options["presence_penalty"] = float(os.environ["OLLAMA_PRESENCE_PENALTY"])
                if "OLLAMA_FREQUENCY_PENALTY" in os.environ:
                    options["frequency_penalty"] = float(os.environ["OLLAMA_FREQUENCY_PENALTY"])
                if "OLLAMA_SEED" in os.environ:
                    options["seed"] = int(os.environ["OLLAMA_SEED"])

                call_start = time.time()
                resp = requests.post(
                    f"{ollama_base}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "think": think,
                        "options": options,
                        "stream": False,
                    },
                    timeout=600,
                )
                resp.raise_for_status()
                result = resp.json()
                call_wall_time = time.time() - call_start

                clean_result = result.get("message", {}).get("content", "")
                thinking_text = result.get("message", {}).get("thinking", "") or ""
                done_reason = result.get("done_reason")
                prompt_tokens = result.get("prompt_eval_count")
                completion_tokens_total = result.get("eval_count")

                # Ollama's native API does not split eval_count between thinking and
                # content tokens. We approximate the split by character-length ratio
                # and label it clearly as an estimate (no exact per-field count exists).
                content_chars = len(clean_result)
                thinking_chars = len(thinking_text)
                total_chars = content_chars + thinking_chars
                if completion_tokens_total and total_chars > 0:
                    completion_tokens_est_content = round(completion_tokens_total * content_chars / total_chars)
                    completion_tokens_est_thinking = completion_tokens_total - completion_tokens_est_content
                else:
                    completion_tokens_est_content = completion_tokens_total
                    completion_tokens_est_thinking = 0

                log_record = {
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "arm": os.environ.get("ARM"),
                    "task": os.environ.get("task"),
                    "instance_idx": os.environ.get("INSTANCE_IDX"),
                    "model": model_name,
                    "think": think,
                    "num_ctx": num_ctx,
                    "max_tokens": max_tokens,
                    "wall_time_sec": round(call_wall_time, 4),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens_total": completion_tokens_total,
                    "completion_tokens_est_content": completion_tokens_est_content,
                    "completion_tokens_est_thinking": completion_tokens_est_thinking,
                    "messages": messages,
                    "content": clean_result,
                    "thinking": thinking_text,
                    "done_reason": done_reason,
                    "content_empty": clean_result.strip() == "",
                    "options": options,
                    "parse_guard": os.environ.get("PARSE_GUARD", "off").strip().lower(),
                    "check_args_fix": os.environ.get("CHECK_ARGS_FIX", "off").strip().lower(),
                    "spy_format": os.environ.get("SPY_FORMAT", "upstream").strip().lower(),
                }
                log_path = os.environ.get("LOG_PATH", "logs/calls.jsonl")
                os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            print(clean_result)
            return clean_result
        except Exception as e:
            print(f"evaluator_construction failed (attempt {ind + 1}/4): {type(e).__name__}: {e}")
            if ind >= 3:
                raise
            ind += 1
            continue


def get_last_processed_index(progress_file):
    """Retrieve the last processed index from the progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            last_index = f.read().strip()
            return int(last_index) if last_index else 0
    else:
        return 0


def update_progress(progress_file, index):
    """Update the last processed index in the progress file."""
    with open(progress_file, 'w', encoding='utf-8') as f:
        f.write(str(index))


def read_jsonline(address):
    not_mark = []
    with open(address, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsonstr = json.loads(jsonstr)
            not_mark.append(jsonstr)
    return not_mark


def read_json(address):
    with open(address, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data


def read_jsonl(address):
    not_mark = []
    with open(address, "r", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            not_mark.append(item)
    return not_mark


def read_csv(address):
    dataset = []
    with open(address, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            dataset.append(row)
    return dataset


def read_tsv(address):
    dataset = []
    with open(address, encoding='utf-8-sig') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for row in tsvreader:
            dataset.append(row)
    return dataset


def read_txt(address, sep):
    dataset = []
    with open(address, 'r', encoding="utf-8") as f:
        for data in f.readlines():
            data = data.replace('\n', '').split(sep)
            dataset.append(data)
    return dataset


def save_jsonline(ls, address):
    for item in ls:
        with open(address, 'a+', encoding='utf-8') as f:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')


def save_json(ls, address):
    json_str = json.dumps(ls, indent=4)
    with open(address, 'w', encoding='utf-8') as json_file:
        json.dump(ls, json_file, ensure_ascii=False, indent=4)


def sort_dic(dic):
    dic = sorted(dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return dic


def create_logger(log_path):
    """
       将日志输出到日志文件和控制台
       """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger
