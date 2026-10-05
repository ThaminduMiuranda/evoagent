import sys
from io import StringIO
import openai
import json
import os
from tqdm import tqdm
import pdb
import logging
import sys
import argparse
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
import numpy as np
import requests
import os
import subprocess
import time
import re
import importlib.util
import os
import pickle
from util_func import *
import sys
from io import StringIO
from threading import Thread, Event
import traceback
from agent_prompt_code import *


def extract_final_answer(answer):
    """Returns the text after 'Final Answer:', or None if the model never wrote it.
    Callers must treat None as a parse failure, not fall back to the raw answer --
    the raw text (meta-reasoning) can contain every target word, which would leak
    the answer into a downstream prompt instead of scoring as a miss.
    """
    if "Final Answer:" not in answer:
        return None
    return answer.split("Final Answer:")[-1].strip()


def message_construction(prompt, model_name):
    if model_name != 'gemini':
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "user", "parts": [prompt]},
        ]
    return messages


def spy_refine_func(ind, question, answer, model_name, data_type):
    i = 0
    answer_list = []
    while i < ind:
        prompt = spy_feedback_agent_prompt.format(question=question, answer=answer)
        messages = message_construction(prompt, model_name)
        feedback_description = evaluator_construction(messages, model_name, question, data_type)

        prompt = spy_self_refine_agent_prompt.format(question=question, answer=answer, feedback=feedback_description)
        messages = message_construction(prompt, model_name)
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


def guess_refine_func(ind, n, question, answer, model_name, data_type):
    i = 0
    answer_list = []
    while i < ind:
        prompt = guess_feedback_agent_prompt.format(question=question, answer=answer)
        messages = message_construction(prompt, model_name)
        feedback_description = evaluator_construction(messages, model_name, question, data_type)

        prompt = guess_self_refine_agent_prompt.format(question=question, answer=answer, feedback=feedback_description,
                                                       n=n)
        messages = message_construction(prompt, model_name)
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


def guess_collaboration_func(ind, n, question, answer, model_name, data_type):
    i = 0
    answer_list = []
    description_ls = []
    while i < ind:
        flag = 0
        while True:
            prompt = guess_meta_agent_prompt.format(question=question, answer=answer,
                                                    description='\n-'.join(description_ls))
            messages = message_construction(prompt, model_name)
            description = evaluator_construction(messages, model_name, question, data_type)

            prompt = check_agent_prompt.format(question=question, description_ls='\n-'.join(description_ls),
                                               description=description)
            messages = message_construction(model_name, prompt)
            check_result = evaluator_construction(messages, model_name, question, data_type)
            print(check_result)

            if 'discard' not in check_result.lower() or flag > 3:
                description_ls.append(description)
                break
            flag += 1

        prompt = guess_multi_agent_prompt.format(question=question, description=description, n=n)
        messages = message_construction(prompt, model_name)
        sub_answer = evaluator_construction(messages, model_name, question, data_type)

        prompt = guess_refine_agent_prompt.format(question=question, description=description,
                                                  old_answer=answer, new_answer=sub_answer, n=n)
        messages = message_construction(prompt, model_name)
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


def spy_collaboration_func(ind, question, answer, model_name, data_type):
    i = 0
    answer_list = []
    description_ls = []
    while i < ind:
        flag = 0
        while True:
            prompt = spy_meta_agent_prompt.format(question=question, answer=answer, description='\n-'.join(description_ls))
            messages = message_construction(prompt, model_name)
            description = evaluator_construction(messages, model_name, question, data_type)

            prompt = check_agent_prompt.format(question=question, description_ls='\n-'.join(description_ls),
                                               description=description)
            messages = message_construction(model_name, prompt)
            check_result = evaluator_construction(messages, model_name, question, data_type)
            print(check_result)
            if 'discard' not in check_result.lower() or flag > 3:
                description_ls.append(description)
                break
            flag += 1

        prompt = spy_multi_agent_prompt.format(question=question, description=description)
        messages = message_construction(prompt, model_name)
        sub_answer = evaluator_construction(messages, model_name, question, data_type)

        prompt = spy_refine_agent_prompt.format(question=question, description=description,
                                                old_answer=answer, new_answer=sub_answer)
        messages = message_construction(prompt, model_name)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemini")
    parser.add_argument("--data_type", type=str, default="gemini")
    parser.add_argument("--method", type=str, default="evoagent")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=31)
    parser.add_argument("--ind", type=int, default=3)
    parser.add_argument("--run_tag", type=str, default="",
                         help="Suffix appended to the progress/result filenames, so arms "
                              "sharing the same model_name+method (e.g. direct w/ and w/o "
                              "thinking) don't collide on the same progress file.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N instances (for smoke tests).")
    args = parser.parse_args()
    model_name = args.model_name
    data_type = args.data_type
    method = args.method
    test_data = read_jsonline('data/codenames_collaborative/codenames_50.jsonl')
    if args.limit is not None:
        test_data = test_data[:args.limit]

    total_files = len(test_data)
    progress_file = f"result/codenames_collaborative_{model_name}_{method}{args.run_tag}.txt"
    start_index = get_last_processed_index(progress_file)
    azure = False
    with tqdm(total=total_files, desc="Processing files", initial=start_index) as pbar:
        for i, data in enumerate(test_data[start_index:], start=start_index):
            os.environ["INSTANCE_IDX"] = str(i)
            # For spymaster
            word_list = data["word_list"]
            target_words = data["target_words"]
            n = len(target_words)
            prompt = f'''Try to find a single word hint that can accurately represent and link the {n} given words: "{target_words}". The key is to select a hint that does not cause confusion with other words from the following word list: {word_list}.
You need to give reasons first and then give the answer with the format: \"Final Answer: <a single word from the word list>\" 
Answer:
'''
            if model_name != 'gemini':
                messages = [
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "user", "parts": [prompt]},
                ]

            if method == "evoagent":
                clean_result = evaluator_construction(messages, model_name, prompt, data_type)
                answer_list, answer = spy_collaboration_func(args.ind, prompt, clean_result, model_name, data_type)
            elif method == "refine":
                clean_result = evaluator_construction(messages, model_name, prompt, data_type)
                answer_list, answer = spy_refine_func(args.ind, prompt, clean_result, model_name, data_type)
            elif method == "direct":
                clean_result = evaluator_construction(messages, model_name, prompt, data_type)
                answer = clean_result
                answer_list = []
            spy_parsed = extract_final_answer(answer)
            spy_parse_failed = spy_parsed is None
            hint_word = spy_parsed if spy_parsed is not None else ""
            data["spy_answer"] = answer
            data["spy_answer_list"] = answer_list
            data["hint_word"] = hint_word
            data["spy_parse_failed"] = spy_parse_failed

            if spy_parse_failed:
                # No parseable hint to give the guesser. Score 0 and move on --
                # do NOT pass the raw reasoning on as the hint, since it can
                # (and in testing, reliably does) contain every target word.
                data["guess_answer"] = None
                data["guess_list"] = []
                data["info"] = {"matched_words": [], "matched_count": 0,
                                "target_count": len(set(w.strip().lower() for w in data["target_words"])),
                                "guess_parse_failed": None}
                with open(os.path.splitext(progress_file)[0] + '.jsonl', 'a+', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                update_progress(progress_file, i + 1)
                pbar.update(1)
                continue

            # For guesser
            word_list = data["word_list"]
            target_words = data["target_words"]
            n = len(target_words)
            prompt = f'''Try to identify the {n} words best associated with the word "{hint_word}" from the following word list: {word_list}.
You need to give reasons first and then give the answer with the format: \"Final Answer: <a comma-separated list of {n} words from the word list>\" 
Answer:
'''
            if model_name != 'gemini':
                messages = [
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "user", "parts": [prompt]},
                ]

            if method == "evoagent":
                clean_result = evaluator_construction(messages, model_name, prompt, data_type)
                answer_list, answer = guess_collaboration_func(args.ind, n, prompt, clean_result, model_name, data_type)
            elif method == "refine":
                clean_result = evaluator_construction(messages, model_name, prompt, data_type)
                answer_list, answer = guess_refine_func(args.ind, n, prompt, clean_result, model_name, data_type)
            elif method == "direct":
                clean_result = evaluator_construction(messages, model_name, prompt, data_type)
                answer = clean_result
                answer_list = []

            target_words = data['target_words']
            target_words = [word.strip().lower() for word in target_words]
            target_words_set = set(target_words)

            guess_parsed = extract_final_answer(answer)
            guess_parse_failed = guess_parsed is None
            if guess_parse_failed:
                # No parseable guess list -- score 0 rather than matching
                # against raw reasoning text (same leak risk as the spy side).
                predicted_words_set = set()
            else:
                predicted_words = [w.strip().replace(".", "").lower() for w in guess_parsed.split(",")]
                predicted_words_set = set(predicted_words)

            common_words = list(predicted_words_set.intersection(target_words_set))
            data["guess_answer"] = answer
            data["guess_list"] = answer_list
            data["guess_parse_failed"] = guess_parse_failed
            data["info"] = {"matched_words": common_words, "matched_count": len(common_words),
                            "target_count": len(target_words_set), "guess_parse_failed": guess_parse_failed}

            with open(os.path.splitext(progress_file)[0] + '.jsonl', 'a+', encoding='utf-8') as f:
                line = json.dumps(data, ensure_ascii=False)
                f.write(line + '\n')

            update_progress(progress_file, i + 1)
            pbar.update(1)
