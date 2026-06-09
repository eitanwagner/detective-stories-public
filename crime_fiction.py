import time
import os

import pandas as pd
import requests
import re
import numpy as np
import sys
from time import sleep
import json
import torch
from textwrap import dedent
import vertexai

# from hgext.fsmonitor.watchmanclient import client
# from sage.ext_data.nbconvert.postprocess import file_name
# from sympy.integrals.meijerint_doc import category
# from torch.ao.nn.quantized.functional import threshold

# from transformers.cache_utils import DynamicCache
# past_key_values = DynamicCache()

from utils import parse_args

args = parse_args()

# NUM_SAMPLES = 10

MODEL_S = "assistant" if args.model_type != "gemini" else "model"
CONTENT_S = "content" if args.model_type != "gemini" else "parts"
CONTENT_S2 = CONTENT_S if args.model_type2 == "" else "content" if args.model_type2 != "gemini" else "parts"
COMPLETION = args.model_type in ["llama", "mistral"] and "-Instruct" not in args.model_name
if args.model_name == "lucaelin/minillama2":
    COMPLETION = False


def exponential_backoff(client, messages, model='gpt-4o-mini-2024-07-18', max_retries=5, base_delay=1, id=0, json=False,
                        multiple=False, max_tokens=1024, model2=False):
    # if "gpt4" in sys.argv:
    #     model = 'gpt-4-turbo-preview'
    # elif "gpt3.5" in sys.argv:
    #     # model = 'gpt-3.5-turbo-16k-0613'
    #     model = 'gpt-3.5-turbo-0125'
    model_name = args.model_name2 if model2 else args.model_name
    if model_name == "o1-mini":
        max_tokens = 64000
        model = "o1-mini-2024-09-12"
        messages = messages[1:]
    elif model_name == "o3-mini":
        max_tokens = 64000
        model = "o3-mini-2025-01-31"
        messages = messages[1:]
    elif model_name == "o1":
        max_tokens = 32000
        model = "o1-preview"
        messages = messages[1:]
    elif model_name == "claude":
        model = "claude-3-opus-20240229"  # change!
    elif model_name == "gpt-4o":
        # model = 'gpt-4o-2024-05-13'
        # model = 'gpt-4o-2024-08-06'
        model = 'gpt-4o-2024-11-20'
    elif model_name == "gpt-4o-mini":
        model = 'gpt-4o-mini-2024-07-18'
    elif model_name == "gpt-5-mini":
        max_tokens = 64000
        model = "gpt-5-mini-2025-08-07"
        # Note: GPT-5 mini does NOT drop system messages like o1/o3
        # So you don't need: messages = messages[1:]
    retries = 0
    while retries < max_retries:
        try:
            if not json:
                if "claude" not in model:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        # max_tokens=max_tokens,
                        max_completion_tokens=max_tokens,
                        n=10 if multiple else 1,
                    )
                else:
                    completion = client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=1.0,
                        system=messages[0]["content"],
                        messages=messages[1:]
                    )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens,
                    # max_tokens=max_tokens,
                    n=10 if multiple else 1,
                )
            return completion
        except Exception as e:
            print(f"Request failed: {str(e)}")
            print(model)
            delay = base_delay * (2 ** retries)
            print(f"Retrying in {delay} seconds...")
            sleep(delay)
            retries += 1
    raise Exception(f"Request failed even after retries. id: {id}")


def exponential_backoff2(chat, message, max_retries=15, base_delay=1, config=None, timeout=120):
    import threading
    retries = 0
    while retries < max_retries:
        try:
            result = [None]
            exc = [None]
            def _call():
                try:
                    result[0] = chat.send_message(message, config=config)
                except Exception as e:
                    exc[0] = e
            t = threading.Thread(target=_call, daemon=True)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                raise TimeoutError(f"Request timed out after {timeout} seconds")
            if exc[0] is not None:
                raise exc[0]
            return result[0]
        except Exception as e:
            print(f"Request failed: {str(e)}")
            delay = base_delay * (2 ** retries)
            print(f"Retrying in {delay} seconds...")
            sleep(delay)
            retries += 1
    raise Exception(f"Request failed even after retries. id: {id}")


def llama_request(pipe, messages):
    # print(messages)
    if COMPLETION:
        return pipe("\n".join(messages), max_new_tokens=500, pad_token_id=pipe.tokenizer.eos_token_id, temperature=1.,
                    do_sample=True)[0][0]['generated_text']
    # return pipe(messages, max_new_tokens=500, pad_token_id=pipe.tokenizer.eos_token_id, temperature=1.,
    #             do_sample=True, past_key_values=past_key_values)[0]['generated_text']
    # return pipe(messages, max_new_tokens=500, pad_token_id=pipe.tokenizer.eos_token_id, temperature=1.,
    #             do_sample=True, cache_implementation="offloaded")[0]['generated_text']
    return pipe(messages, max_new_tokens=500, pad_token_id=pipe.tokenizer.eos_token_id, temperature=1.,
                do_sample=True)[0]['generated_text']


def llama_request2(pipe, messages):
    # Check if pipe is a Vertex AI GenerativeModel
    if hasattr(pipe, 'generate_content'):
        # Vertex AI path
        from vertexai.generative_models import Content, Part

        # Convert messages to Vertex AI format
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            # Chat format - convert to Vertex AI Contents
            contents = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                content = msg.get("content", "")
                contents.append(Content(role=role, parts=[Part.from_text(content)]))

            response = pipe.generate_content(
                contents,
                generation_config={
                    "max_output_tokens": 500,
                    "temperature": 1.0,
                }
            )
            return [{"role": "assistant", "content": response.text}]
        else:
            # Completion format
            response = pipe.generate_content(
                "\n".join(messages),
                generation_config={
                    "max_output_tokens": 500,
                    "temperature": 1.0,
                }
            )
            return response.text

    # Original local pipeline path
    if COMPLETION:
        return pipe("\n".join(messages), max_new_tokens=500, pad_token_id=pipe.tokenizer.eos_token_id, temperature=1.,
                    do_sample=True)[0][0]['generated_text']
    return pipe(messages, max_new_tokens=500, pad_token_id=pipe.tokenizer.eos_token_id, temperature=1.,
                do_sample=True)[0]['generated_text']


def llama_likelihood(model, tokenizer, messages=None):
    """
    Calcluate the log-likelihood for text by the given causal model
    :param model:
    :param tokenizer:
    :param text:
    :return:
    """

    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt')
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    loss = out.loss
    log_likelihood = -loss.item() * input_ids.size(1)  # Convert per-token to sequence log-likelihood
    print(out.logits.size())
    return log_likelihood


# *************************

def make_sbs_messages_real(paragraphs, suspects=None, story_length=20, c_hint=False, assumptions=""):
    ms = []

    # if args.model_type not in ["llama", "mistral", "gemini"]:
    if args.model_type not in ["anthropic", "gemini"]:
        ms = [{"role": "system", "content": "You are a story writer."}]
    m1 = dedent(f"""\
    We will write together a detective story. I will give you paragraphs and you will complete the story. 
    """)
    if args.story_type == "edwin":
        m1 = dedent(f"""\
        We will complete together a story started by Charles Dickens. 
        I will give you parts and you will complete the story part by part. Notice that each part can consist of multiple paragraphs.
        The goal is not to replicate Dickens' intention but rather to complete a plausible story.
        """)
        if args.surprising:
            m1 = m1 + "You should try to make the ending surprising."
    elif args.story_type == "poirot":
        print("using a prompt to avoid memorized stories")
        # m1 = m1 + dedent("""
        # This will be a Hercule Poirot story and you should write it in Agatha Christie's style.
        # Importantly, even if part of the story is similar or identical to a real story, the continuation should be based on memory---you should create a new story.
        # The identity of the true culprit that will be revealed should be based only on the clues that appeared. It can turn out to be any of the suspects (if the clues lead to him) --- NOT necessarily the one in stories you previously saw.
        # """)
        m1 = m1 + dedent("""
        This will be a Hercule Poirot story and you should write it in Agatha Christie's style.
        Importantly, even if part of the story is similar or identical to a real story, the continuation should NOT be based ONLY on memory. That is, you should continue the story as if you never saw it. 
        The ending (and the true culprit) CAN be the true ending (if that is what sampling for the "continuation distribution" gave you), but it CAN also be different if that is what came up from the process.
        Your goal is to simulate the generation process of Agatha Christie's stories, taking into account the style and clues that were revealed.
        """)
    elif args.story_type == "sherlock":
        print("using a prompt to avoid memorized stories")
        # m1 = m1 + dedent("""
        # This will be a Sherlock Holmes story and you should write it in Arthur Conan Doyle's style.
        # Importantly, even if part of the story is similar or identical to a real story, the continuation should be based on memory---you should create a new story.
        # The identity of the true culprit that will be revealed should be based only on the clues that appeared. It can turn out to be any of the suspects (if the clues lead to him) --- NOT necessarily the one in stories you previously saw.
        # """)
        m1 = m1 + dedent("""
        This will be a Sherlock Holmes story and you should write it in Arthur Conan Doyle's style.
        Importantly, even if part of the story is similar or identical to a real story, the continuation should NOT be based ONLY on memory. That is, you should continue the story as if you never saw it. 
        The ending (and the true culprit) CAN be the true ending (if that is what sampling for the "continuation distribution" gave you), but it CAN also be different if that is what came up from the process.
        Your goal is to simulate the generation process of Arthur Conan Doyle's stories, taking into account the style and clues that were revealed.
        """)

    m1 = m1 + dedent(f"""            
    The true culprit must be one of following suspects:
    {", ".join(suspects)}. """)

    m1 = m1 + dedent("""
    I will give you the paragraph number and only then will you write the paragraph itself. In each step you will generate a single paragraph -- do NOT write multiple paragraphs in a single step. 
    Make sure to reveal the true culprit by the end of the story.
    Do not add numbers to the paragraphs.
    Be focused on the story writing without extra explanations.
    """)

    segment_name = "Paragraph" if args.story_type != "edwin" else "Part"
    m2 = dedent(f"""\
    Understood. Please provide the {segment_name.lower()} number.
    """)
    ms.append({"role": "user", CONTENT_S: m1})
    ms.append({"role": MODEL_S, CONTENT_S: m2})

    if paragraphs is not None:
        for i, p in enumerate(paragraphs):
            ms.append({"role": "user", CONTENT_S: f"Now generate {segment_name} {i + 1} out of {story_length}"})
            if i == story_length - 1:
                ms[-1][CONTENT_S] = ms[-1][CONTENT_S] + f" (the last {segment_name.lower()} in the story)"
            ms.append({"role": MODEL_S, CONTENT_S: p})
    return ms


def make_sbs_messages(paragraphs, suspects=None, story_length=20, c_hint=False, assumptions=""):
    ms = []
    c_hint_t = ""
    if c_hint:
        c_hint_t = dedent("""\
        This will be draft that you will be able to revise later. Your goal will be to use clues that point to the real outcome but do not explicitly reveal the truth until desired. 
        You can add comments that you will remove when revising. Mark these comments (e.g., with brackets) for easy removal.""")
    # s_hint = "Your goal is to make the story misleading in a way that, until the true culprit is revealed, the evidence will point to a different suspect. Once revealed, predicting the culprit should be simple.\n" \
    #     if args.s_hint else ""
    s_hint = ""

    a_hint2 = dedent("""
                    To achieve the goals, I will first ask you to write some assumptions that you expect the naive reader to hold and can be debunked at the revelation. In the story itself, you should generate clues that will be interpreted wrongly (pointing to the distractor) with the wrong assumptions, and once the truth is revealed, the clues will be interpreted correctly.  
                    These assumptions will be visible at the time of writing but will not be part of the final story. After writing the assumptions I will ask you to generate the actual paragraphs of the story.
                    """)
    a_hint = dedent("""
                    To achieve the goals, I will first ask you to write some assumptions that you expect the naive reader to hold and will prevent him from understanding the true nature of the clues until the revelation. 
                    In the story itself, you should generate clues that will be interpreted wrongly (pointing to the distractor) with the wrong assumptions, and once the truth is revealed, the clues will be interpreted correctly, pointing to the true culprit.  
                    These assumptions will be visible at the time of writing but will not be part of the final story. After writing the assumptions I will ask you to generate the actual paragraphs of the story.
                    """)
    if args.plan_outline:
        a_hint = dedent("""
                        To achieve the goals, I will first ask you to write some an outline of the story. 
                        This can include a general plan of what will happen, like the type of crime and motive, as well as details that will be revealed throughout the story, like the clues that will appear.
                        """)

    # if args.model_type not in ["llama", "mistral", "gemini"]:
    if args.model_type not in ["anthropic", "gemini"]:
        ms = [{"role": "system", "content": "You are a story writer."}]
    m1 = dedent(f"""\
    We will generate a misleading detective story step by step. There will be {story_length} paragraphs in total. 
    I will ask you to generate one paragraph at a time.
    """) + s_hint + c_hint_t

    if suspects is not None:
        m1 = m1 + dedent(f"""            
        The true culprit must be one of following suspects:
        {", ".join(suspects)}. """)
    else:
        m1 = m1 + dedent("""
        At least four characters should be introduced in the story and one of them will be the culprit. 
        """)

    if args.story_type not in ["sherlock", "poirot", "rivals"]:
        m1 = m1 + dedent("""\
        The story must also have a distracting character that should be suspected as the true culprit.
        The clues should be misleading and point to the distracting character until the end of the story when the truth is discovered. Make sure that the clues are consistent with the true outcome. 
        The distracting character must be one of the four suspects.
        """)
    elif args.story_type == "poirot":
        m1 = m1 + dedent("""This is a Hercule Poirot story and you should write it in Agatha Christie's style.
        Importantly, even if part of the story is similar or identical to a real story, the continuation should be based on memory---you should create a new story.
        The identity of the true culprit that will be revealed should be based only on the clues that appeared. It can turn out to be any of the suspects (if the clues lead to him). 
        """)
    elif args.story_type == "sherlock":
        m1 = m1 + dedent("""This is a Sherlock Holmes story and you should write it in Arthur Conan Doyle's style.
        Importantly, even if part of the story is similar or identical to a real story, the continuation should be based on memory---you should create a new story.
        The identity of the true culprit that will be revealed should be based only on the clues that appeared. It can turn out to be any of the suspects (if the clues lead to him). 
        """)

    if ((args.plan_assumptions or args.plan_outline) and paragraphs is None) or assumptions != "":
        m1 = m1 + a_hint

    m1 = m1 + dedent("""
    I will give you the paragraph number and only then will you write the paragraph itself. In each step you will generate a single paragraph -- do NOT write multiple paragraphs in a single step. 
    Make sure to reveal the true culprit by the end of the story.
    Do not add numbers to the paragraphs.
    Be focused on the story writing without extra explanations.
    """)

    if ((args.plan_assumptions or args.plan_outline) and paragraphs is None) or assumptions != "":
        m2 = "\nUnderstood.\n"
    else:
        m2 = dedent("""\
        Understood. Please provide the paragraph number.
        """)
    ms.append({"role": "user", CONTENT_S: m1})
    ms.append({"role": MODEL_S, CONTENT_S: m2})

    if assumptions != "":
        ms.append({"role": "user", CONTENT_S: "Now write assumptions that you expect the naive reader to hold."})
        # ms.append({"role": "user", CONTENT_S: "Now write assumptions that you expect the naive reader to hold and can be debunked."})
        ms.append({"role": MODEL_S, CONTENT_S: assumptions})

    if paragraphs is not None:
        for i, p in enumerate(paragraphs):
            ms.append({"role": "user", CONTENT_S: f"Now generate Paragraph {i + 1} out of {story_length}"})
            if i == story_length - 1:
                ms[-1][CONTENT_S] = ms[-1][CONTENT_S] + " (the last paragraph in the story)"
            ms.append({"role": MODEL_S, CONTENT_S: p})
    return ms


def generate_step_by_step(client=None, client2=None, story_length=20, paragraphs=None, suspects=None,
                          return_culprit=False, stop_length=None, ms=None, clues=None, return_first=False):
    """
    Create the story step by step
    :param client:
    :param suspects:
    :param story_length:
    :return:
    """
    if client2 is None:
        client2 = client
    model_type = args.model_type
    model_type2 = args.model_type2 if args.model_type2 != "" else model_type

    if ms is None:
        ms = make_sbs_messages(paragraphs, suspects=suspects, story_length=story_length, c_hint=args.c_hint)
    _story_length = story_length
    if stop_length is not None:
        _story_length = stop_length

    n_paragraphs = 0
    if paragraphs is not None:
        n_paragraphs = len(paragraphs)

    continuation = ""
    if clues is None or len(clues) == 0:
        clues = []
        use_clues = False
    else:
        use_clues = True

    if model_type == "gemini":
        # chat = client.start_chat(history=ms)
        _ms = [{'role': m['role'], "parts": [{"text": m["parts"]}]} for m in ms]
        chat = client.chats.create(model=args.model_name, history=_ms)
        from google.genai import types
        config = types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_budget=args.thinking_budget),
            safety_settings = [types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                               types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                               types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                               types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),]
        )


    assumptions = ""
    if (args.plan_assumptions or args.plan_outline) and paragraphs is None:
        print(f"Generating assumptions")
        # !!!!!!
        if args.plan_assumptions:
            ms.append({"role": "user",
                       CONTENT_S: "Now write assumptions that you expect the naive reader to hold and can be debunked."})
        else:
            ms.append({"role": "user", CONTENT_S: "Now write the outline."})
        if model_type != "gemini":
            if args.model_type not in ["llama", "mistral"]:  # openai and anthropic
                completion = exponential_backoff(client, ms)
                ms.append(
                    {"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
            else:
                if "70B" in args.model_name:
                    completion = llama_request2(pipe=client, messages=messages)[-1]
                else:
                    completion = llama_request(pipe=client, messages=messages)[-1]
                # completion = llama_request(pipe=client, messages=ms)[-1]
                ms.append(completion)
            assumptions = ms[-1]["content"]
        else:  # gemini
            response = exponential_backoff2(chat,
                                            "Now write assumptions that you expect the naive reader to hold and can be debunked.", config=config)
            ms.append({"role": "model", "parts": response.text})
            assumptions = ms[-1]["parts"]
        print(ms)

    first_para = ""
    segment_name = "Paragraph" if args.story_type != "edwin" else "Part"
    for i in range(n_paragraphs, _story_length):
        print(f"Generating {segment_name} {i + 1} out of {_story_length}")
        sys.stdout.flush()

        content = f"Now generate {segment_name} {i + 1} out of {_story_length}"
        if i == _story_length - 1:
            content = content + f" (the last {segment_name.lower()} in the story)"
        ms.append({"role": "user", CONTENT_S: content})
        if use_clues:
            # import re
            ms[-1][CONTENT_S] = content + ".\nThe following clues should appear in the new paragraph:\n" + re.sub(
                r'\d+\. ', '* ', clues[i])

        if model_type != "gemini":
            # if i == _story_length - 1:
            #     ms[-1]["content"] = ms[-1]["content"] + " (the last paragraph in the story)"
            if model_type not in ["llama", "mistral"]:
                completion = exponential_backoff(client, ms)
                ms[-1][CONTENT_S] = content
                ms.append(
                    {"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
            else:
                # print(ms)
                if "70B" in args.model_name:
                    completion = llama_request2(pipe=client, messages=ms)[-1]
                else:
                    completion = llama_request(pipe=client, messages=ms)[-1]
                # completion = llama_request(pipe=client, messages=ms)[-1]
                # print(completion)
                ms[-1][CONTENT_S] = content
                ms.append(completion)
            new_para = ms[-1]["content"]
            # continuation = continuation + ms[-1]["content"] + ("\n\n###\n\n" if i < _story_length - 1 else "")
        else:
            for r in range(15):
                response = exponential_backoff2(chat, f"Now generate {segment_name} {i + 1} out of {_story_length}", config=config)
                # response = chat.send_message(f"Now generate Paragraph {i + 1} out of {_story_length}")
                if response.text is None:
                    print(f"response.text is None. Sleeping for {2**r} seconds")
                    sleep(2**r)
                    continue
                ms[-1][CONTENT_S] = content
                ms.append({"role": "model", "parts": response.text})
                new_para = ms[-1]["parts"]
                break

        continuation = continuation + new_para + ("\n\n###\n\n" if i < _story_length - 1 else "")
        if i == n_paragraphs:
            first_para = new_para

        if args.clues:
            prompt = dedent("Give me a list of clues that were presented in this last paragraph."
                            "I want you to describe the clues as theoretical possibilities without reference to specific characters.\n"
                            "Make sure to be faithful to the paragraph and to list only clues that were explicitly mentioned. Do not add new clues or clues from previous paragraphs.\n"
                            "The clues should be hard evidence only, like things the detective found, and not interpretations or conclusions.\n"
                            "Give the list of clues only, without any additional explanations.")

            if model_type != "gemini":
                # ms.append(
                #     {"role": "user", CONTENT_S: prompt})
                if model_type not in ["llama", "mistral"]:
                    completion = exponential_backoff(client, ms + [{"role": "user", CONTENT_S: prompt}])
                    # ms.append(
                    #     {"role": completion.choices[0].message.role,
                    #      "content": completion.choices[0].message.content})
                    clues.append(completion.choices[0].message.content)
                else:
                    if "70B" in args.model_name:
                        completion = llama_request2(pipe=client, messages=ms)[-1]
                    else:
                        completion = llama_request(pipe=client, messages=ms)[-1]
                    # completion = llama_request(pipe=client, messages=ms)[-1]
                    # ms.append(completion)
                    clues.append(completion[CONTENT_S])
            else:
                response = exponential_backoff2(chat, prompt)
                # ms.append({"role": "model", "parts": response.text})
                clues.append(response.text)

    if return_culprit:
        m = dedent(f"""\
        Now give me the number of the culprit (from the list of suspects). 
        The list of suspects is:
        {", ".join(suspects)}. 

        The answer should be the number only.    
        """)
        if "1B" in args.model_name:
            m = dedent(f"""\
            Now give me the number of the culprit (from the list of suspects). 
            The list of suspects is:
            1. {suspects[0]}, 2. {suspects[1]}, 3. {suspects[2]}, and 4. {suspects[3]}.
    
            The answer should be the number only (from 1 to 4). 
            """)
            print("using a new culprit prompt for 1B model")

        if model_type2 != "gemini":
            if model_type == "gemini":
                # convert to openai format
                ms = [{"role": ("assistant" if m["role"] == "model" else m["role"]), "content": m["parts"]} for m in ms]
            ms.append({"role": "user", "content": m})
            if model_type2 not in ["llama", "mistral"]:
                completion = exponential_backoff(client2, ms, model2=True)
                ms.append(
                    {"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
            else:
                completion = llama_request(pipe=client2, messages=ms)[-1]
                ms.append(completion)
            c = re.findall(r"\d", ms[-1]["content"])
        else:
            # chat = client2.start_chat(history=ms)
            # response = chat.send_message(m)
            # ms.append({"role": "model", "parts": response.text})
            # c = re.findall(r"\d", ms[-1]["parts"])
            _ms2 = [{'role': _m['role'], "parts": [{"text": _m["parts"]}]} for _m in ms]
            chat2 = client2.chats.create(model=args.model_name2, history=_ms2)
            from google.genai import types
            config2 = types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=1.0,
                safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
                ]
            )
            response = exponential_backoff2(chat2, m, config=config2)
            ms.append({"role": "model", "parts": response.text})
            c = re.findall(r"\d", ms[-1]["parts"])

        if len(c) == 0:
            culprit = -1
            print("!!!! No number found !!!!")
        else:
            culprit = int(c[0])
        return ms, continuation, culprit, assumptions, first_para

    return ms, continuation, chat if args.model_type == "gemini" else None, assumptions, clues


def make_revise_m1(story, suspects, story_length=20):
    # r_hint = "Your goal is to increase the likelihood of the clues in the story, given the outcome, while keeping the outcome itself unpredictable until the truth is revealed.\n" \
    #     if "r_hint" in sys.argv else ""
    s_hint = "Your goal is to make the story misleading in a way that, until the true culprit is revealed, the evidence will point to a different suspect. After the revelation, the evidence in whole should point to the real culprit. You should modify the clues to better maintain these two objectives.\n" \
        if args.s_hint else ""
    c_hint = "In your revision you should remove comments that should be not be visible in the final story.\n" \
        if args.c_hint else ""
    m1 = dedent(f"""
    I'm giving you a misleading detective story with four characters, one of them the culprit and one of them a distractor. 
    I want you to revise the story to make it more coherent with respect to the outcome.\n""") + \
         s_hint + \
         dedent(f"""You will be able to add, remove or change details in any part of the story but make sure to keep the number of paragraphs the same. 
    The suspects from the original story: {", ".join(suspects[:-1])} and {suspects[-1]} must appear in the revised story. The true and distracting culprit must remain the same.\n""") + \
         c_hint + \
         dedent(f"""
    I will ask you to generate one paragraph at a time. I will give you the paragraph number and only then will you write the paragraph. You will also shortly describe the revision.
    The total number of paragraphs will be {story_length} (the same as the original).
    Be focused on the story writing without extra explanations.
    """)
    if args.r_json and args.model_type not in ["llama", "mistral", "gemini"]:
        m1 = m1 + dedent("""    
    For each paragraph, you should return a JSON dictionary with the following format:
    ```json
    {
    "paragraph": <text>,
    "revisions": <short description of the revisions made>
    }
    Make sure to have these fields in the JSON. Put an empty string if no revisions were made.
    ```
    """)
    m1 = m1 + \
         dedent("""

    The story is:

    ## BEGINNING OF STORY ##

    """) + story + \
         dedent("""

    ## END OF STORY ##

    """)
    return m1


def revise_step_by_step(story="", client=None, client2=None, num_revisions=1, story_length=20, suspects=None):
    """
    Revise the story step by step
    :param client:
    :param story: The story to revise
    :param num_revisions: Number of revisions
    :param story_length: Number of paragraphs in the story
    """

    m2 = f"""
    Understood. Please provide the paragraph number.
    """

    descriptions = []
    continuations = [story]
    ms = []
    for j in range(num_revisions):
        _descriptions = []
        if args.model_type not in ["anthropic", "gemini"]:
            ms = [{"role": "system", "content": "You are a story writer."}]

        m1 = make_revise_m1(continuations[-1], suspects, story_length=story_length)
        ms.append({"role": "user", CONTENT_S: m1})
        ms.append({"role": MODEL_S, CONTENT_S: m2})
        print("\n*********************")
        print(f"Revision {j}")
        continuation = ""

        if args.model_type == "gemini":
            chat = client.start_chat(history=ms)

        for i in range(0, story_length):
            print(f"Generating Paragraph {i + 1} out of {story_length}")
            sys.stdout.flush()

            if args.r_json and args.model_type not in ["llama", "mistral", "gemini"]:
                ms.append({"role": "user",
                           CONTENT_S: f"Now generate the JSON (with \"paragraph\" and \"revisions\") for Paragraph {i + 1} out of {story_length}"})
            else:
                ms.append({"role": "user", CONTENT_S: f"Now generate Paragraph {i + 1} out of {story_length}"})

            if args.model_type != "gemini":
                if i == story_length - 1:
                    ms[-1]["content"] = ms[-1]["content"] + " (the last paragraph in the story)"
                if args.model_type not in ["llama", "mistral"]:
                    completion = exponential_backoff(client, ms, json=args.r_json)
                    ms.append(
                        {"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
                else:
                    if "70B" in args.model_name:
                        completion = llama_request2(pipe=client, messages=ms)[-1]
                    else:
                        completion = llama_request(pipe=client, messages=ms)[-1]
                    # completion = llama_request(pipe=client, messages=ms)[-1]
                    ms.append(completion)
                # continuation = continuation + ms[-1][CONTENT_S] + ("\n\n###\n\n" if i < story_length - 1 else "")
            else:
                response = exponential_backoff2(chat, f"Now generate Paragraph {i + 1} out of {story_length}")
                ms.append({"role": "model", "parts": response.text})
                # if args.r_json:
                #     try:
                #         j_content = json.loads("{" + response.text.split("{")[-1].split("}")[0] + "}")
                #         _descriptions.append(j_content["revisions"])
                #         continuation = continuation + j_content["paragraph"] + ("\n\n###\n\n" if i < story_length - 1 else "")
                #     except Exception as e:
                #         print("Result was not a JSON")
                #         print(e)
                #         print(response.text)
                #         return [], [], []
                # else:
                #     continuation = continuation + ms[-1][CONTENT_S] + ("\n\n###\n\n" if i < story_length - 1 else "")

            if args.r_json and args.model_type not in ["llama", "mistral", "gemini"]:
                j_content = ms[-1][CONTENT_S]

                # j_content1 = j_content.split("paragraph")

                if j_content.find("}") == -1:
                    j_content = j_content + "}"

                j_content = j_content.replace("\'revisions\': ", "\"revisions\": ")
                j_content = j_content.replace("\'paragraph\': ", "\"paragraph\": ")
                _j = j_content.split("\"revisions\"")
                try:
                    if "\"revisions\"" in j_content and _j[0].rsplit("\"")[-1].find(",") == -1:  # if there is no comma
                        j_content = _j[0].strip() + ",\n\"revisions\"" + _j[1]
                    if "\"revisions\"" in j_content and _j[0].rfind("\"") < _j[0].rfind("\'"):
                        j_content = _j[0][:_j[0].rfind("\'")] + "\",\n\"revisions\"" + _j[1]
                    _j = j_content.split("\"revisions\"")
                    j_content = json.loads("{" + j_content.split("{", 1)[1].rsplit("}", 1)[0] + "}")

                except Exception as e:
                    try:
                        j_content = "{\"paragraph\": \"" + \
                                    _j[0][:_j[0].rfind("\"")].split("\"paragraph\": \"")[1].replace("\"", "'") + \
                                    "\",\n\"revisions\"" + _j[1]
                        # j_content = json.loads("{" + j_content.split("{", 1)[1].rsplit("}", 1)[0] + "}")
                        j_content = json.loads(j_content.rsplit("}", 1)[0] + "}")

                    except Exception as e:
                        print("Result was not a JSON2")
                        print(e)
                        print(j_content)
                        return [], [], []

                    print("JSON WAS FIXED")
                    print(j_content)

                if "revisions" in j_content:
                    _descriptions.append(j_content["revisions"])
                else:
                    _descriptions.append("")
                _descriptions.append(j_content["revisions"])
                continuation = continuation + j_content["paragraph"] + (
                    "\n\n###\n\n" if i < story_length - 1 else "")
            else:
                continuation = continuation + ms[-1][CONTENT_S] + ("\n\n###\n\n" if i < story_length - 1 else "")

                if args.r_json:
                    if args.model_type != "gemini":
                        ms.append({"role": "user",
                                   CONTENT_S: "Describe the revision that you made for the last paragaraph (if any)."})
                        if args.model_type not in ["llama", "mistral"]:
                            completion = exponential_backoff(client, ms)
                            ms.append(
                                {"role": completion.choices[0].message.role,
                                 "content": completion.choices[0].message.content})
                        else:
                            if "70B" in args.model_name:
                                completion = llama_request2(pipe=client, messages=ms)[-1]
                            else:
                                completion = llama_request(pipe=client, messages=ms)[-1]
                            # completion = llama_request(pipe=client, messages=ms)[-1]
                            ms.append(completion)
                            _descriptions.append(ms[-1][CONTENT_S])
                        # continuation = continuation + ms[-1][CONTENT_S] + ("\n\n###\n\n" if i < story_length - 1 else "")
                    else:
                        response = exponential_backoff2(chat,
                                                        "Describe the revision that you made for the last paragaraph (if any).")
                        ms.append({"role": "model", "parts": response.text})
                        _descriptions.append(response.text)

        continuations.append(continuation)
        descriptions.append(_descriptions)
        # story = continuation

    return ms, continuations[-1], descriptions


# *************************

def ask_probs(client=None, client2=None, story="", print_output=True, suspects=None, distractor=False, strategy="", prev_stories=None):
    if client2 is None:
        client2 = client
    model_type = args.model_type2 if args.model_type2 != "" else args.model_type
    model_name = args.model_name2 if args.model_name2 != "" else args.model_name

    if suspects is None:  # only for the first time, and the full story is given
        m = dedent("""\
        I want you to give me likelihood estimates for the true and distracting culprit identities in the following story.
        I want lists of four prediction probabilities, a value for each of four suspects (representing the likelihood that the suspect is the true or distracting culprit). Even if a suspect is completely ruled out, he should still be included (and assigned a low probability).

        Give me a JSON dictionary with the probabilities. Make sure to follow the exact format in the example (and include all fields). Give me only the JSON. Do not put comments in the JSON.
        Put the four suspects by the order that they were introduced in the story. Use full names (e.g., Sam Smith) with no additional description (like Dr.). There should be exactly four suspects.

        For example, if it's clear that A is the distracting culprit and B is the most likely true culprit, then return something like:
        ```json
        {
        "suspects": ["A", "B", "C", "D"],
        "probabilities": [0.0, 0.9, 0.05, 0.05],
        "distractor_probabilities": [1.0, 0.0, 0.0, 0.0]
         }
        ```
        """)
    elif distractor:
        m = dedent(f"""\
        I want you to give me likelihood estimates for the true and distracting culprit identities in the following story.
        I want lists of four prediction probabilities, a value for each of four suspects (representing the likelihood that the suspect is the true or distracting culprit). Even if a suspect is completely ruled out, he should still be included (and assigned a low probability).

        Give me a JSON dictionary with the probabilities. Make sure to follow the exact format in the example. Give me only the JSON. Do not put comments in the JSON.
        The suspects are: {', '.join(suspects)}. Please do not change the identity and order of suspects.""") + \
            dedent("""

        For example, if it's clear that A is the distracting culprit and B is the most likely true culprit, then return something like:
        ```json
        {
        "suspects": ["A", "B", "C", "D"],
        "probabilities": [0.0, 0.9, 0.05, 0.05],
        "distractor_probabilities": [1.0, 0.0, 0.0, 0.0]
         }
        ```
        """)
    else:
        m = dedent(f"""\
        I want you to predict the culprit's identity, based on the (possibly partial) story that will follow.
        The prediction should be the likelihood of each outcome based only on the explicit evidence that was reported.
        I want a list of prediction probabilities, a value for each of the suspects (representing the likelihood that the suspect is the true culprit). Even if a suspect is completely ruled out, he should still be included (and assigned a low probability).

        Give me a JSON dictionary with the probabilities. Make sure to follow the exact format in the example. Give me only the JSON. Do not put comments in the JSON.
        The suspects are: {', '.join(suspects)}. Please do not change the identity and order of suspects.""") + \
            dedent("""

        For example, if it's clear that A is the culprit then return:
        ```json
        {
        "suspects": ["A", "B", "C", "D"],
        "probabilities": [1.0, 0.0, 0.0, 0.0]
        }
        ```
        """)

    m_naive = "\nI want you to give probability estimations as if the events in the story are real, ignoring the interest of the writer. You should estimate the most likely truth, even if is boring.\n"
    m_final2 = (
        "\nI want you to give probability estimations for what will be revealed (or was already revealed) at the end of story. That is, the probability for each culprit is the probability that the story will be continued in way that ends up with this character being the culprit.\n"
        "Take into account that the story has a writer who might have an objective to mislead the reader.\n"
        "Your prediction should be based on the story that is revealed on NOT based on prior knowledge of this story.\n")
    m_ffinal = (
        "\nI want you to give probability estimations for what will be revealed (or was already revealed) at the end of story. That is, the probability for each culprit is the probability that the story will be continued in way that ends up with this character being the culprit.\n"
        "Your might have prior knowledge of this story and its ending. In this case you SHOULD use the knowledge.\n")
    m_final = (
        f"\nI want you to give probability estimations for what will be revealed (or was already revealed) at the end of story. That is, the probability for each culprit is the probability that the story will be continued in way that ends up with this character being the culprit.\n"
        f"I Note that the story is generated paragraph by paragraph by a language model, for a total of {args.num_paragraphs} paragraphs.\n"
        f"The generating model is instructed to introduce at least four characters, one of which is the true culprit and one is a distractor.\n"
        f"The clues are instructed to be misleading and point to the distracting character until the end of the story when the truth is discovered.\n")
    if strategy == "naive":
        m = m + m_naive
    elif "final" in strategy:
        if args.story_type not in ["poirot", "sherlock", "rivals", "edwin"]:
            m = m + m_final
        else:
            m = m + m_final2
    # elif strategy == "ffinal":
    #     pass

    if "ffinal" in strategy:
        if args.story_type not in ["poirot", "sherlock", "rivals", "edwin"]:
            m = m + dedent("\nTry your best to deduce the culprit's identity. "
                       "Use any knowledge at your disposal, including previous stories you've seen\n")
        else:
            m = m + dedent("\nTry your best to deduce the culprit's identity. "
                       "Use any knowledge at your disposal, including previous stories you've seen. "
                       "However, do not use the knowledge of the specific story that you are currently analyzing, if you have it.\n")

    if strategy in ["ffinal2", "ffinal3", "ffinal2a", "ffinal3a"]:
        if strategy == "ffinal3":
            _prev_stories = prev_stories[:3]
        elif strategy == "ffinal2":
            _prev_stories = prev_stories[:]
        elif strategy == "ffinal2a":
            _prev_stories = prev_stories[:2]
        elif strategy == "ffinal3a":
            _prev_stories = prev_stories[:5]
        m = (m + \
        dedent("""\nI'm also giving you some previous full stories written by this model. Use them to deduce the culprit's identity if relevant.
        
        #######################
        ## PREVIOUS STORIES ##
        """) + "\n".join([
        dedent(f"""
        ## BEGINNING OF PREVIOUS STORY {pi}##

        """) + p_story + \
            dedent(f"""

        ## END OF PREVIOUS STORY {pi}##""") for pi, p_story in enumerate(_prev_stories)])
        + \
        dedent("""
        
        ## END OF PREVIOUS STORIES ##
        #############################\n\n
        """))

    if len(strategy) > 0 and  strategy[-1] == "a":
        m = m + \
            dedent("""
            ---------------------------------------------------------------------------------
            
        Now I'll give you the (possibly partial) story for which you must predict the culprit.
        The story is:
    
        ## BEGINNING OF (PARTIAL) STORY FOR PREDICTION ##
    
        """) + story + \
            dedent("""
    
        ## END OF (PARTIAL) STORY FOR PREDICTION ##
    
        """)
    elif len(strategy) > 0:
        m = m + \
            dedent("""

        The story is:

        ## BEGINNING OF STORY ##

        """) + story + \
            dedent("""

        ## END OF STORY ##

        """)
    elif story:
        m = m + \
            dedent("""

        The story is:

        ## BEGINNING OF STORY ##

        """) + story + \
            dedent("""

        ## END OF STORY ##

        """)

    messages = []
    if model_type in ["openai", "mistral", "llama"]:
        messages.append({"role": "system", "content": "You are a story writer."})

    messages.append({"role": "user", CONTENT_S2: m})

    # if model_type == "gemini":
    #     chat = client2.start_chat(history=messages)
    if model_type == "gemini":
        from google.genai import types
        _ms = [{'role': m['role'], "parts": [{"text": m[CONTENT_S2]}]} for m in messages]
        chat = client2.chats.create(model=model_name, history=_ms)
        config = types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=1.0,
            safety_settings=[
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ]
        )

    if model_type not in ["llama", "mistral", "gemini", "anthropic"]:  # openai
        completion = exponential_backoff(client2, messages, json="o1" not in model_name and "o3" not in model_name, model2=True)
        # completion = exponential_backoff(client2, messages, json=False, model2=True)
        messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    elif model_type in ["llama", "mistral"]:
        completion = llama_request(pipe=client2, messages=messages)[-1]
        messages.append(completion)
    # elif model_type == "gemini":
    #     response = chat.send_message(m)
    #     messages.append({"role": "model", "parts": response.text})
    elif model_type == "gemini":
        response = exponential_backoff2(chat, m, config=config)
        messages.append({"role": "model", "parts": response.text})
    else:  # anthropic
        completion = exponential_backoff(client2, messages, model2=True)
        messages.append(
            {"role": completion.role, "content": completion.content[0].text})
    if print_output:
        print(messages)

    try:
        json_content = messages[-1]["content"] if model_type != "gemini" else messages[-1]["parts"]
        if model_type in ["llama", "mistral", "gemini", "anthropic"] or "o1" in model_name or "o3" in model_name:
            # remove comments
            json_content = "\n".join([l.split("//")[0].split(" # ")[0] for l in json_content.splitlines()])
            json_content = "{" + json_content.split("{", 1)[1].rsplit("}", 1)[0].replace("0.,", "0.0") + "}"
        d = json.loads(json_content)
    except Exception as e:
        print("Result was not a JSON")
        print(e)
        print(messages)
        return ([], []) if (not distractor and suspects is not None) else ([], [], [])
    if "distractor_probabilities" in d:
        return d["suspects"], d["probabilities"], d["distractor_probabilities"]
    elif suspects is None:
        return d["suspects"], d["probabilities"], None
    return d["suspects"], d["probabilities"]


def fill_para(client=None, client2=None, story="", paras=None):
    if client2 is None:
        client2 = client
    model_type = args.model_type2 if args.model_type2 != "" else args.model_type
    model_name = args.model_name2 if args.model_name2 != "" else args.model_name

    m = dedent("""\
    I will give you a story with a missing paragraph (marked by "[MISSING]") and six options for filling it. 

    Give me a list of probabilities for the paragraph that can fill in the missing spot. 
    I stress that the goal is not to predict what was in the spot but rather to answer about what makes sense given the actual ending.""")

    if args.test_clues_type != "":
        m = m + "Notice that some paragraphs might be hidden (marked by [HIDDEN]). I will NOT ask you about filling those paragraphs, only the [MISSING] one.\n"

    m = m + dedent("""
    For example, if the second options is the best, the first is also possible and the others make very little sense, then you answer should look like: 
    ```json
    {
    "options": ["a", "b", "c", "d", "e", "f"],
    "probabilities": [0.25, 0.55, 0.05, 0.05, 0.05, 0.05]
    }
    ```

    Your response should be the JSON dictionary only, with no additional text.
    """)

    m = m + \
        dedent("""

    The story is:

    ## BEGINNING OF STORY ##

    """) + story + \
        dedent("""

    ## END OF STORY ##

    """)
    m = m + \
        dedent("""
    The optional paragraphs are:

    """) + "\n\n".join(l + par for l, par in zip(["a. ", "b. ", "c. ", "d. ", "e. ", "f. "], paras)) + "\n\n"

    messages = []
    if model_type in ["openai", "mistral", "llama"]:
        messages.append({"role": "system", "content": "You are a story writer."})

    # if model_type == "gemini":
    #     chat = client2.start_chat(history=messages)
    if model_type == "gemini":
        from google.genai import types
        _ms = [{'role': m['role'], "parts": [{"text": m[CONTENT_S2]}]} for m in messages]
        chat = client2.chats.create(model=model_name, history=_ms)
        config = types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=1.0,
            safety_settings=[
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ]
        )

    messages.append({"role": "user", CONTENT_S2: m})

    if model_type not in ["llama", "mistral", "gemini", "anthropic"]:  # openai
        completion = exponential_backoff(client2, messages, json="o1" not in model_name and "o3" not in model_name, model2=True)
        messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    elif model_type in ["llama", "mistral"]:
        completion = llama_request(pipe=client2, messages=messages)[-1]
        messages.append(completion)
    # elif model_type == "gemini":
    #     response = chat.send_message(m)
    #     messages.append({"role": "model", "parts": response.text})
    elif model_type == "gemini":
        response = exponential_backoff2(chat, m, config=config)
        messages.append({"role": "model", "parts": response.text})
    else:  # anthropic
        completion = exponential_backoff(client2, messages, model2=True)
        messages.append(
            {"role": completion.role, "content": completion.content[0].text})

    try:
        json_content = messages[-1]["content"] if model_type != "gemini" else messages[-1]["parts"]
        if model_type in ["llama", "mistral", "gemini", "anthropic"] or "o1" in model_name or "3" in model_name:
            # remove comments
            json_content = "\n".join([l.split("//")[0].split(" # ")[0] for l in json_content.splitlines()])
            json_content = "{" + json_content.split("{", 1)[1].rsplit("}", 1)[0].replace("0.,", "0.0") + "}"
        d = json.loads(json_content)
    except Exception as e:
        print("Result was not a JSON")
        print(e)
        print(messages)
        return ([], [])
    return d["options"], d["probabilities"]


def choose_culprit(client=None, client2=None, story="", print_output=True, suspects=None, multiple=False,
                   predict_strategy="predict_now"):
    """
    Choose the culprit directly (without continuing the story)
    :param client:
    :param messages:
    :return: The number representing the culprit
    """

    m_s = dedent(f"""
    The list of suspects is: 
    {suspects}.
    """)

    m_now = dedent("""
    Now predict the currently most probable culprit (from the same list). The prediction should be based only on the \
    (possibly partial) evidence that was already revealed.
    The answer should be the number only (the index in the list, from 1 to 4).
    """)
    m_final = dedent("""
    Now predict the most probable culprit (from the same list) for the final outcome. Take into account that there might \
    be extra twists and reveals in the story. Notice the location within the story.
    The answer should be the number only (the index in the list, from 1 to 4).
    """)
    m_outline = dedent("""
    I want you to predict the most probable culprit (from the same list) for the final outcome. For this, you should \
    generate an outline for the continuation of the story, based on what was revealed. 
    The outline should be short but it should conclude the story in a way that makes the most probable culprit clear.
    Take into account that there might be extra twists and reveals in the story. Notice the location within the story.
    Along with the outline, give me the number of the culprit (the index in the list given earlier, from 1 to 4).

    The response should be in JSON format as follows:
    ```json
    {
    "outline": <outline>
    "culprit": <number>
    }
    ``` 
    """)

    m_ = dedent(f"""\
                Here is a story:

                ## BEGINNING OF STORY ##

                """) + \
         story + \
         dedent("""

        ## END OF STORY ##

        """)

    messages = []
    if args.model_type == "openai":
        messages.append({"role": "system", "content": "You are a story writer."})

    if args.model_type == "gemini":
        chat = client.start_chat(history=messages)

    culprits = []

    return_json = False
    max_tokens = 1024
    if predict_strategy == "predict_now":
        max_tokens = 128
        m = m_ + m_s + m_now
    elif predict_strategy == "predict_final":
        max_tokens = 128
        m = m_ + m_s + m_final
    elif predict_strategy == "predict_outline":
        if len(messages) > 75:
            max_tokens = 300
        return_json = True
        m = m_ + m_s + m_outline

    if args.model_type not in ["llama", "mistral", "gemini"]:
        messages.append({"role": "user", "content": m})
        completion = exponential_backoff(client, messages, multiple=multiple, json=return_json,
                                         max_tokens=max_tokens)
        if "claude" not in sys.argv:
            messages.append(
                {"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
        else:
            messages.append(
                {"role": completion.role, "content": completion.content[0].text})
    elif args.model_type == "gemini":
        # response = chat.send_message(m)
        response = exponential_backoff2(chat, m)
        messages.append({"role": "user", "parts": m})
        messages.append({"role": "model", "parts": response.text})
    else:
        messages.append({"role": "user", "content": m})
        if "70B" in args.model_name:
            completion = llama_request2(pipe=client, messages=messages)[-1]
        else:
            completion = llama_request(pipe=client, messages=messages)[-1]
        messages.append(completion)

    if print_output:
        print(messages)
    r = range(10) if multiple else range(1)
    for j in r:
        try:
            content = messages[-1][CONTENT_S]
            if return_json:
                if args.model_type in ["llama", "mistral", "gemini"]:
                    # content = content.split("```json")[-1].split("```")[0].replace("0.,", "0.0")
                    content = "{" + content.split("{", 1)[1].rsplit("}", 1)[0].replace("0.,", "0.0") + "}"
                d = json.loads(content)
                culprits.append(int(d["culprit"]))
            else:
                ds = np.array([content.find(d) for d in ["1", "2", "3", "4"]])
                if sum(ds >= 0) > 1:
                    print("!!!!!!!!!!!!!!!!!!")
                    print("More than one number")
                    print(content)
                    print("!!!!!!!!!!!!!!!!!!")
                content = str(np.argmax(ds) + 1)
                culprits.append(int(re.findall(r"\d", content)[0]))

        except Exception as e:
            print("Result was not a number")
            print(e)
    return culprits


def load_from_dict(id=0, type="sherlock", revisions=0, print_data=False, return_others=False):
    """
    Load a story from the dictionary
    :param id:
    :param type:
    :param candidates:
    :return:
    """
    name = ""
    np = ""
    print(type)
    # type = args.story_type
    real_types = ["sherlock", "poirot", "rivals", "edwin"]
    if type in real_types:
        name = f"{type}_w_suspects"
    elif type in ["gpt-4o", "gpt-4o-mini", "o1-mini", "o1", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash", "gemini-2.5-pro",
                  "Llama-3.1-8B-Instruct", "gemini-3-flash-preview", "gemini-3.1-pro-preview",
                  "Llama-3.1-70B-Instruct", "Mixtral-8x7B-Instruct-v0.1",
                  "Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct",
                  "Llama-3.3-70B-Instruct", "Mixtral-8x7B-v0.1", "Mistral-Nemo-Instruct-2407"]:
        name = f"{type}_w_suspects{'_pa' if args.plan_assumptions else ''}{'_po' if args.plan_outline else ''}_sbs_d"
        np = args.num_paragraphs
    elif "gemini-2.5-flash" in type or "gemini-2.5-pro" in type:
        name = f"{type}_w_suspects{'_pa' if args.plan_assumptions else ''}{'_po' if args.plan_outline else ''}_sbs_d"
        np = args.num_paragraphs

    if revisions > 0:
        name = name + f"_revised{revisions}"
    file_name_ = args.base_path + f"stories/{name}{'_c' if args.c_hint and args.load_from_dict else ''}{'_clues' if args.clues and args.load_from_dict else ''}{'_s' if args.s_hint and args.load_from_dict else ''}{np}{'_w_details2' if (args.test_clues_type != '' and '1B' not in args.model_name) else '_w_details' if (args.test_clues_type != '') else ''}.json"
    with open(file_name_,
            "r") as file:
        stories = json.load(file)

    # print(stories)

    # if dict convert to list
    if type in ["sherlock", "poirot", "edwin"]:
        stories = list(stories.values())
        stories = [(s["name"], s) for s in stories]
    else:
        stories = [(s["name"], s) for s in stories]
    stories.sort()

    if print_data:
        print(stories[id][0])
        print(stories[id][1])
    # return stories[id][1]["text"], stories[id][1]["suspects"]
    if return_others:
        other_texts = [stories[i][1]["text"] for i in range(len(stories)) if i != id]
        return stories[id][1], other_texts
    return stories[id][1]


def split_into_paragraphs(text, min_paragraph_length=200, numbered=False):
    # Split text into paragraphs by two or more new lines
    paragraphs = text.split('\n\n')

    # Combine short paragraphs
    combined_paragraphs = []
    current_paragraph = paragraphs[0]

    for paragraph in paragraphs[1:]:
        # Check if the paragraph is short (less than min_paragraph_length characters)
        if len(paragraph) < min_paragraph_length:
            # Combine the short paragraph with the current paragraph
            current_paragraph = f"{current_paragraph}\n\n{paragraph}"
        else:
            # Add the current paragraph to the list and move to the next paragraph
            combined_paragraphs.append(current_paragraph)
            current_paragraph = paragraph

    # Append the last paragraph
    combined_paragraphs.append(current_paragraph)
    if numbered:
        return [str(i + 1) + ". " + c for i, c in enumerate(combined_paragraphs)]
    return combined_paragraphs


def _get_client2(client2=False):
    model_type, model_name = args.model_type, args.model_name
    if client2:
        model_type, model_name = args.model_type2, args.model_name2
        if model_type == "":
            return None

    if model_type in ["llama", "mistral"]:
        # Use Vertex AI for Llama models
        if model_type == "llama":

            from vertexai.generative_models import GenerativeModel
            import os

            # Set GOOGLE_APPLICATION_CREDENTIALS in your environment before running.
            # Initialize Vertex AI
            vertexai.init(
                project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )

            # Map model names to Vertex AI model IDs
            vertex_model_map = {
                "Llama-3.1-8B-Instruct": "publishers/meta/models/llama-3.1-8b-instruct-maas",
                "Llama-3.1-70B-Instruct": "publishers/meta/models/llama-3.1-70b-instruct-maas",
                "Llama-3.2-1B-Instruct": "publishers/meta/models/llama-3.2-1b-instruct-maas",
                "Llama-3.2-3B-Instruct": "publishers/meta/models/llama-3.2-3b-instruct-maas",
                "Llama-3.3-70B-Instruct": "publishers/meta/models/llama-3.3-70b-instruct-maas",
            }

            vertex_model_id = vertex_model_map.get(model_name, f"meta/{model_name.lower()}")
            client = GenerativeModel(vertex_model_id)
            print(f"Using Vertex AI model: {vertex_model_id}")
        else:
            # Keep Mistral models as local for now (or add similar Vertex AI logic)
            from transformers import pipeline
            import torch
            model_name = f"mistralai/{model_name}"
            client = pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16},
                              device_map="auto")
            print(model_name)
    elif model_type == "openai":
        from openai import OpenAI
        from openai_client import get_client
        client = get_client()
    elif model_type == "anthropic":
        import anthropic
        from openai_client import get_anthropic_client
        client = get_anthropic_client()
    elif model_type == "gemini":
        from openai_client import get_google_client
        client = get_google_client()

    return client


def _get_client(client2=False):
    model_type, model_name = args.model_type, args.model_name
    if client2:
        model_type, model_name = args.model_type2, args.model_name2
        if model_type == "":
            return None

    if model_type in ["llama", "mistral"]:
        # Mistral-Nemo-Instruct-2407, Mistral-Nemo-Base-2407, Mixtral-8x7B-Instruct-v0.1, Mixtral-8x7B-v0.1, Mistral-7B-Instruct-v0.3, Mistral-7B-v0.3
        # Llama-3.1-70B-Instruct, Llama-3.1-70B, Llama-3.1-8B-Instruct, Llama-3.1-8B
        from transformers import pipeline
        import torch
        if args.model_type == "llama":
            # if "3.1" in model_name:
            #     model_name = f"meta-llama/Meta-{model_name}"
            # else:
            model_name = f"meta-llama/{model_name}"
        else:
            model_name = f"mistralai/{model_name}"
        client = pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16},
                          device_map="auto")
        print(model_name)
    elif model_type == "openai":
        from openai import OpenAI
        from openai_client import get_client
        client = get_client()
    elif model_type == "anthropic":
        import anthropic
        from openai_client import get_anthropic_client
        client = get_anthropic_client()
    elif model_type == "gemini":
        # import google.generativeai as genai
        # from openai_client import configure_gemini
        # configure_gemini()
        from openai_client import get_google_client
        client = get_google_client()
        # from google.generativeai.types import HarmCategory, HarmBlockThreshold
        # client = genai.GenerativeModel(model_name=model_name,
        #                                system_instruction="You are a story writer.",
        #                                safety_settings={
        #                                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        #                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        #                                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        #                                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, },
        #                                generation_config=genai.GenerationConfig(max_output_tokens=3000,
        #                                                                         temperature=1.0))

    return client


# *************************

def main():
    print_output = False

    type = args.story_type if args.story_type != "" else args.model_name
    if "gemini-2.5" in args.model_name:
        type = type + "_tb" + str(args.thinking_budget)
    id = args.id

    if args.test_clues_type == "":
        if "70B" in args.model_name:
            print("Using llama_request2")
            client = _get_client2()
        else:
            client = _get_client()  # notice that for gemini it's a chat model
    else:
        client = None
    client2 = _get_client(client2=True)
    print(client)
    print(client2)

    # if args.test:
    #     print(client.model.config)
    #     print(client.model.generation_config.cache_config)
    #     print(client.model.generation_config.cache_implementation)
    #     return

    # generate stories
    if args.make_dict:
        story_length = args.num_paragraphs
        story_list = []
        for i in range(20):
            print(i)
            sys.stdout.flush()

            if len(story_list) >= 10:
                break
            s_dict = {}
            ms, continuation, _, assumptions, clues = generate_step_by_step(client, client2, story_length=story_length)

            suspects, cs, d_cs = ask_probs(client, client2, story=continuation)
            print(suspects)
            print(cs)
            if d_cs is None or len(suspects) != 4 or max(cs) <= 0.5 or max(d_cs) <= 0.5 or "None" in suspects or \
                    np.argmax(cs) == np.argmax(d_cs):
                continue

            s_dict["suspects"] = suspects
            s_dict["assumptions"] = assumptions
            s_dict["probs"] = cs
            s_dict["distractor_probs"] = d_cs
            s_dict["text"] = continuation
            s_dict["clues"] = clues
            s_dict["name"] = f"Story {i} - {type} - Step by step"
            if "c_hint" in sys.argv:
                s_dict["name"] = s_dict["name"] + " with comments"
            story_list.append(s_dict)

        with open(
                args.base_path + f"stories/{type}_w_suspects{'_pa' if args.plan_assumptions else ''}{'_po' if args.plan_outline else ''}_sbs_d{'_c' if args.c_hint else ''}{'_clues' if args.clues else ''}{story_length}.json",
                "w") as file:
            json.dump(story_list, file)

    # make revisions
    revisions = args.revisions
    if args.make_revisions:
        story_list = []

        for i in range(10):
            s_dict = {}
            d = load_from_dict(type=type, id=i, revisions=0)
            continuation, suspects = d["text"], d["suspects"]

            for j in range(3):
                try:
                    _ms, _continuation, _descriptions = revise_step_by_step(continuation, client, client2,
                                                                            num_revisions=revisions,
                                                                            suspects=suspects,
                                                                            story_length=args.num_paragraphs)
                except Exception as e:
                    print(e)
                    print(f"Story {i} failed in revision {j}")
                    continue
                if len(_continuation) > 0:
                    ms, continuation, descriptions = _ms, _continuation, _descriptions
                    break

            for j in range(3):
                try:
                    suspects, cs, d_cs = ask_probs(client, client2, story=continuation, suspects=suspects,
                                                   distractor=True)
                except Exception as e:
                    print(e)
                    print(f"Story {i} failed in asking probabilities")
                    continue
                if len(suspects) > 0:
                    break

            print(suspects)
            print(cs)
            s_dict["suspects"] = suspects
            s_dict["r_descriptions"] = descriptions
            s_dict["probs"] = cs
            s_dict["distractor_probs"] = d_cs
            s_dict["text"] = continuation
            s_dict["name"] = f"Story {i} - {type} - Step by step - revised_{revisions}"
            if "c_hint" in sys.argv:
                s_dict["name"] = s_dict["name"] + " made with comments"
            story_list.append(s_dict)

        with open(
                args.base_path + f"stories/{type}_w_suspects{'_pa' if args.plan_assumptions else ''}{'_po' if args.plan_outline else ''}_sbs_d_revised{revisions}{'_c' if args.c_hint else ''}{'_clues' if args.clues else ''}{'_s' if args.s_hint else ''}{args.num_paragraphs}.json",
                "w") as file:
            json.dump(story_list, file)

    if args.load_from_dict:
        if args.estimation_strategies2 or args.estimation_strategies3:
            book_d, others = load_from_dict(type=type, id=id, revisions=revisions, print_data=True, return_others=True)
            book_d["other_texts"] = others
        else:
            book_d = load_from_dict(type=type, id=id, revisions=revisions, print_data=True)
        book_d["id"] = id
        text, suspects = book_d["text"], book_d["suspects"]
        print("loaded story")

        assumptions = ""
        if args.plan_assumptions2:
            assumptions = book_d["assumptions"]

        opening_paragraphs = []
        if type in ["poirot", "sherlock", "rivals"]:
            paragraphs = split_into_paragraphs(text)
            paragraphs = [p for p in paragraphs if len(p) > 0]
        else:
            paragraphs = text.split("\n\n###\n\n")
            paragraphs = [p for p in paragraphs if len(p) > 0]

        story_length = len(paragraphs)
        if type == "edwin":
            paragraphs = paragraphs[1:]
            story_length += 20

        print("Lengths:")
        print(len(text))
        print(len(paragraphs))

        if type in ["poirot", "sherlock", "rivals", "edwin"]:
            ms = make_sbs_messages_real(paragraphs, suspects=suspects, story_length=story_length,
                                        assumptions=assumptions)
        else:
            ms = make_sbs_messages(paragraphs, suspects=suspects, story_length=story_length, assumptions=assumptions)
        # convert to a beginning of a story that can be longer
        test_strategies(ms, paragraphs, client, client2, print_output, book_d=book_d)


def test_strategies(ms, paragraphs, client, client2, print_output, book_d=None):
    suspects = book_d["suspects"]
    clues = book_d.get("clues", None)
    r_probs = book_d["probs"] if "probs" in book_d else book_d["real"]
    if args.test_clues_type != "":
        rev_point = book_d['true_culprit_paragraph']

        # load the results for this story with sampling
        # mn2 = "o1-mini" if args.model_name2 == "o3-mini" else args.model_name2
        mn2 = "o3-mini" if '1B' not in args.model_name else "o1-mini"
        if args.story_type in ["poirot", "sherlock", "rivals", "edwin"]:
            with open(
                    args.base_path + f"results/-mn {args.model_name} -mn2 {mn2} -st {args.story_type} --id {args.id} -ld -mt {args.model_type} -mt2 {args.model_type2} -es -tc{' -pa' if args.plan_assumptions else ''}{' -po' if args.plan_outline else ''}.json",
                    "r") as file:
                p_results = json.load(file)
                para_fill_all = p_results['para_fill']
                p_ids = p_results['ids']
        else:
            file_name_ = args.base_path + f"results/-mn {args.model_name} -mn2 {mn2} -st {args.story_type} -ld -mt {args.model_type} -mt2 {args.model_type2}{' -np ' if args.num_paragraphs == 25 else ''}{args.num_paragraphs if args.num_paragraphs == 25 else ''} --id {args.id} -es -tc{' -pa' if args.plan_assumptions else ''}{' -po' if args.plan_outline else ''}{' -nsamp 20' if (args.num_samples == 20) else ''}.json"
            if args.model_name in ["gemini-2.5-flash"]:
                file_name_ = args.base_path + f"results/-mn {args.model_name} -tb {args.thinking_budget} -mn2 {mn2} -st {args.story_type} -ld -mt {args.model_type} -mt2 {args.model_type2}{' -np ' if args.num_paragraphs == 25 else ''}{args.num_paragraphs if args.num_paragraphs == 25 else ''} --id {args.id} -es -tc{' -pa' if args.plan_assumptions else ''}{' -po' if args.plan_outline else ''}{' -nsamp 20' if (args.num_samples == 20) else ''}.json"
            with open(file_name_) as file:
                p_results = json.load(file)
                para_fill_all = p_results['para_fill']
                p_ids = p_results['ids']

    # new_cs = []
    new_cs = {0: [], 1: [], 2: []}
    new_cs2 = []
    new_cs3 = {0: [], 1: [], 2: []}
    new_cs4 = []

    to_test = list(range(len(paragraphs)))
    if 25 <= len(paragraphs) < 50:
        to_test = list(range(0, len(paragraphs) // 2, 2)) + list(range(len(paragraphs) // 2, len(paragraphs), 1))
    elif len(paragraphs) >= 50:
        to_test = list(range(0, len(paragraphs) // 2, 3)) + list(range(len(paragraphs) // 2, len(paragraphs), 2))
    if args.less_sampling:
        # take only each second element in to_test (and the last)
        to_test = to_test[:-1:2] + [to_test[-1]]

    story_length = len(paragraphs)
    if args.story_type == "edwin":
        # to_test = list(range(0, len(paragraphs) // 2, 2)) + list(range(len(paragraphs) // 2, len(paragraphs), 4)) + [len(paragraphs)]
        # to_test = [10, 12, 13, 14, 16, 18, 20, 21, 22]
        to_test = [1, 7, 13, 18, 22]
        story_length = len(paragraphs) + 20

    strategies = ["predict_now"]
    e_strategies = ["direct"]
    if args.predict_strategies:
        strategies = ["predict_now", "predict_final", "predict_outline"]
    if args.estimation_strategies:
        if args.story_type not in ["poirot", "sherlock", "rivals", "edwin"]:
            e_strategies = ["direct", "naive", "final"]
        else:
            e_strategies = ["ffinal", "naive", "final"]
        strategies = []
    elif args.estimation_strategies2:
        e_strategies = ["ffinal", "ffinal2", "ffinal3"]
        strategies = []
    elif args.estimation_strategies3:
        e_strategies = ["ffinala", "ffinal2a", "ffinal3a"]
        strategies = []
    if args.test_clues_type != "":
        e_strategies = []
        if args.test_clues_type in ["none", "all", "only_before"]:
            e_strategies = ["", "naive", ""]
        strategies = []

    name = " ".join(sys.argv[1:])
    checkpoint_path = args.base_path + f"results/{name}.checkpoint.json"
    completed_is = set()

    para_ids = []
    para_fill = []

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "r") as _f:
            _ckpt = json.load(_f)
        para_ids = _ckpt["para_ids"]
        new_cs = {int(k): v for k, v in _ckpt["new_cs"].items()}
        new_cs2 = _ckpt["new_cs2"]
        new_cs3 = {int(k): v for k, v in _ckpt["new_cs3"].items()}
        para_fill = _ckpt["para_fill"]
        completed_is = set(_ckpt["completed_is"])
        print(f"Resuming: {len(completed_is)} paragraphs already done")

    for i, p in enumerate(paragraphs):
        print("\nstep:" + str(i + 1))
        print(p)
        if i not in to_test:
            continue
        if i in completed_is:
            print(f"Skipping paragraph {i + 1} (checkpoint)")
            continue

        print("num paragraph for estimation: " + str(i + 1))

        para_ids.append(i + 1)
        story = "\n\n###\n\n".join(paragraphs[:para_ids[-1]])

        # system_length = 1 if args.model_type not in ["llama", "mistral", "gemini", "anthropic"] else 0
        system_length = 1 if args.model_type not in ["gemini", "anthropic"] else 0
        num_messages = 2 * (2 + i + int(args.plan_assumptions2)) + system_length

        for k, s in enumerate(e_strategies):
            if s == "":
                continue
            print("Probability estimates:")
            print(s)
            _suspects, _cs = ask_probs(client, client2, story, print_output=print_output, suspects=suspects, strategy=s, prev_stories=book_d["other_texts"] if book_d and "other_texts" in book_d else None)
            print(suspects)
            print(_suspects)
            print(_cs)
            new_cs[k].append(_cs)

        print("Sampling ratio:")
        cs2 = np.zeros(len(suspects))
        r = range(args.num_samples)
        if args.test_clues_type == "":
            _para_fill = {"first_paras": [], "options": [], "probabilities": [], "culprits": []}
        else:
            _para_fill = para_fill_all[p_ids.index(i + 1)]
        if args.no_sampling or args.test_clues_type != "":
            r = range(0)
        for _ in r:
            if i == 0:
                cs2 = np.ones(len(suspects))
                print("skipping first")
                break
            _ms, story, c, _, first_para = generate_step_by_step(client, client2, suspects=suspects,
                                                                 paragraphs=range(i + 1),
                                                                 story_length=story_length,
                                                                 return_culprit=True, ms=list(ms[:num_messages]),
                                                                 clues=clues)
            _para_fill["first_paras"].append(first_para)
            _para_fill["culprits"].append(c)

            c = [c]
            for _c in c:
                if _c > len(suspects) or _c < 1:
                    print("bad c")
                else:
                    cs2[_c - 1] += 1

            # sleep for a short time to avoid rate limits
            if args.num_samples > 5 and args.model_type == "gemini":
                time.sleep(1)

        new_cs2.append(list(cs2 / (cs2.sum() if cs2.sum() > 0 else 1)))
        print(list(cs2))

        if (args.test_clues or args.test_clues_type) and i + 1 < story_length:
            m_paras = paragraphs.copy()

            if args.test_clues_type == "no_end":
                #remove also one paragraph before the revelation because sometimes it already there
                m_paras[int(rev_point)-1:] = ["[HIDDEN]"] * (len(m_paras) - int(rev_point) + 1)
            elif args.test_clues_type == "before_and_end":
                m_paras[i + 1:int(rev_point)-1] = ["[HIDDEN]"] * (int(rev_point)-i-2)
            elif args.test_clues_type == "only_end":
                m_paras[i + 1:int(rev_point)-1] = ["[HIDDEN]"] * (int(rev_point)-i-2)
                m_paras[:i + 1] = ["[HIDDEN]"] * (i + 1)
            elif args.test_clues_type == "only_before":
                m_paras[i + 1:] = ["[HIDDEN]"] * (len(m_paras)-i-1)
            elif args.test_clues_type == "none":
                m_paras = ["[HIDDEN]"] * len(m_paras)
            elif args.test_clues_type == "all":
                pass
            m_paras[i + 1] = "[MISSING]"
            masked_story = "\n\n###\n\n".join(m_paras)

            if args.test_clues_type == "":
                _para_fill["first_paras"].append(paragraphs[i + 1])
                _para_fill["culprits"].append(int(np.argmax(r_probs)) + 1)
            _para_fill["options"], _para_fill["probabilities"] = fill_para(client, client2, story=masked_story,
                                                                           paras=_para_fill["first_paras"][-6:])
            print(_para_fill)
            para_fill.append(_para_fill)
        else:
            _para_fill["first_paras"].append("")
            _para_fill["culprits"].append(int(np.argmax(r_probs)) + 1)
            _para_fill["options"], _para_fill["probabilities"] = ["a", "b", "c", "d", "e", "f"], [0.] * 6
            para_fill.append(_para_fill)

        print("Predict ratio:")
        r = range(0)
        # r = range(5)
        multiple = False
        # r = range(1)
        # multiple = True

        for k, s in enumerate(strategies):
            print("Strategy: " + s)
            cs3 = np.zeros(len(suspects))
            for j in r:
                # for j in range(1):
                c = choose_culprit(client, client2, story, suspects=suspects, print_output=print_output,
                                   multiple=multiple, predict_strategy=s)
                for _c in c:
                    if _c is None or _c > len(suspects) or _c < 1:
                        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! bad c !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    else:
                        cs3[_c - 1] += 1
            new_cs3[k].append(list(cs3 / (cs3.sum() if cs3.sum() > 0 else 1)))
            print(list(cs3))

        completed_is.add(i)
        with open(checkpoint_path, "w") as _f:
            json.dump({"para_ids": para_ids, "new_cs": new_cs, "new_cs2": new_cs2,
                       "new_cs3": new_cs3, "para_fill": para_fill,
                       "completed_is": list(completed_is)}, _f)
        print(f"Checkpoint saved ({len(completed_is)}/{len(to_test)} paragraphs)")

    print("\n\n***********************************")
    print("Results:\n")
    print(sys.argv)
    print("Direct probability estimates:")
    print(f"Estimation Strategies: {e_strategies}")
    for k, v in new_cs.items():
        print(k)
        print(v)
    # print(new_cs)
    print("Sampling ratios:")
    print(new_cs2)
    print("Predict ratios:")
    print(f"Strategies: {strategies}")
    for k, v in new_cs3.items():
        print(k)
        print(v)
    print("Random choice ratios:")
    print(new_cs4)

    print("Results in dict:\n")
    d = {"paragraphs": paragraphs, "ids": para_ids, "direct": new_cs, "sampling": new_cs2, "predict": new_cs3,
         "suspects": suspects, "real_probs": book_d.get("probs", []),
         "distractor_probs": book_d.get("distractor_probs", []),
         "para_fill": para_fill}
    print(d)
    print("***********************************\n\n\n")

    with open(args.base_path + f"results/{name}.json", "w") as file:
        json.dump(d, file)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return


if __name__ == "__main__":
    print(sys.argv)
    # print the number of available GPUs
    print("Number of GPUs:")
    print(torch.cuda.device_count())

    main()
    print("Done")

# # remove descriptions that are in the text
# with open("stories/" + "gemini-1.5-pro_w_suspects_sbs_d_revised1_s30.json", "r") as file:
#     stories = json.load(file)
# for s in stories:
#     text = s["text"]
#     paras = text.split("\n\n###\n\n")
#     _paras = [p.split("\n\n## Revision")[0].split("\n\n**Revision")[0].split("\n\n*Revision")[0] for p in paras]
#     s["text"] = "\n\n###\n\n".join(_paras)
# with open("stories/" + "gemini-1.5-pro_w_suspects_sbs_d_revised1_s30.json", "r") as file:
#     stories = json.load(file)