
from time import sleep
import json
import os
from crime_fiction import load_from_dict, exponential_backoff2
from collections import Counter
import numpy as np

def exponential_backoff(client, messages, max_retries=5, base_delay=1):

    max_tokens = 64000
    # model = "o1-mini-2024-09-12"
    # model = "o3-mini-2025-01-31"
    model = "gpt-5-mini-2025-08-07"
    # messages = messages[1:]

    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                # max_tokens=max_tokens,
                max_completion_tokens=max_tokens,
                n=1,
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

def load_from_dict(type="sherlock", save=False, stories=None, w_stats=False, po=False):
    """
    Load stories from the dictionary
    :param type:
    :return:
    """
    name = ""
    np = ""
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    # type = args.story_type
    if type in ["sherlock", "poirot", "rivals", "edwin"]:
        name = f"{type}_w_suspects"
    elif type in ["gpt-4o", "gpt-4o-mini", "o1-mini", "o1", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash_tb-1", "gemini-2.5-flash_tb0", "gemini-2.5-pro_tb-1", "Llama-3.1-8B-Instruct",
                  "Llama-3.1-70B-Instruct", "Mixtral-8x7B-Instruct-v0.1", "gemini-3-flash-preview", "gemini-3.1-pro-preview",
                  "gemini-3-flash", "gemini-3-pro",
                  "Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.3-70B-Instruct", "Mixtral-8x7B-v0.1", "Mistral-Nemo-Instruct-2407"]:
        name = f"{type}_w_suspects{'_po' if po else ''}_sbs_d"
        # name = f"{type}_w_suspects_po_sbs_d"
        np = 25 if type not in ["o1-mini"] else 30

    if save:
        with open(base_path + f"stories/{name}{np}_w_details2.json",
                "w") as file:
            json.dump(stories, file)
    else:
        import os
        if w_stats:
            path2 = base_path + f"stories/{name}{np}_w_details2.json"
            path1 = base_path + f"stories/{name}{np}_w_details.json"
            path = path2 if os.path.exists(path2) else path1
        else:
            path = base_path + f"stories/{name}{np}.json"
        with open(path, "r") as file:
            stories = json.load(file)

    # # if dict convert to list
    # if type in ["sherlock", "poirot", "edwin"]:
    #     stories = list(stories.items())
    # else:
    #     stories = [(s["name"], s) for s in stories]
    # stories.sort()

    return stories

def ask_stats(client, story):
    from textwrap import dedent
    m = dedent(""" I will give you a detective story with numbered paragraphs.
    
    I want you to ask the following questions:
    1. What is the first name of the victim?
    2. What is the last name of the victim? 
    3. What is the first name of the culprit?
    4. What is the last name of the culprit? 
    5. What is the first name of the detective?
    6. What is the last name of the detective? 
    In all the questions about names, the answer should be one word only. Do not add prefixes like "Mr." or "Dr.". If irrelevant give an empty string.
    
    7. Is the victim male or female?
    8. Is the culprit male or female?
    9. Is the detective male or female? 
    In these questions, the answer should be "male", "female" or "other".
    
    10. What is the method of the crime? 
    The answer must be one of the following list: ["stabbing", "poisoning", "shooting", "burglary", "suicide", "kidnapping", "other"]
    
    11. What is the motive of the crime?
    The answer must be one of the following list: ["financial", "revenge", "mistake", "other"]
    
    12. In what paragraph is the true culprit discovered?
    13. In what paragraph is the distracting culprit established as the prime suspect?
    14. By what paragraph are the main suspects introduced? For example, if the last suspect is introduces in paragraph number 3 then return 3.
    In the question the answer should be a single number. If irrelevant give an empty string.
    
    You will give me a JSON dictionary with the answers for the questions. 
    The format must be:
    ```json
    {
    "victim_first_name": "",
    "victim_last_name": "",
    "culprit_first_name": "",
    "culprit_last_name": "",
    "detective_first_name": "",
    "detective_last_name": "",
    "victim_gender": "",
    "culprit_gender": "",
    "detective_gender": "",
    "method": "",
    "motive: "",
    "true_culprit_paragraph: ,
    "distracting_culprit_paragraph: ,
    "last_introduction_paragraph: ,
    }
    ```
    
    """)
    m2 = dedent(""" I will give you a detective story with numbered paragraphs.
    
    I want you to do the following:
    Given the truth that was revealed, I want you to reconstruct the true crime by order with as many details as possible. 
    Importantly, the true crime is only the events that are actually relevant to the crime (in terms of means, motive, and opportunity) and happened before and during the crime. 
    Events that are part of the investigation process, or distracting from the true crime should NOT be included.
    Then, give a list, for each paragraph in the given story, describing what part of the true crime is introduced in that paragraph. 
    If the details were already introduced, do not mark it again. If no details are introduced, give an empty string for the paragraph.
    
    You will give me a JSON dictionary. The format must be:
    ```json
    {
    "reconstructed story": "",
    "explanation": []
    }
    ```
    
    """)

    m = m + \
    dedent("""

    The story is:

    ## BEGINNING OF STORY ##

    """) + story + \
    dedent("""

    ## END OF STORY ##

    """)

    messages = []
    messages.append({"role": "user", "content": m})
    completion = exponential_backoff(client, messages)
    messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})

    json_content = messages[-1]["content"]
    json_content = "\n".join([l.split("//")[0].split(" # ")[0] for l in json_content.splitlines()])
    json_content = "{" + json_content.split("{", 1)[1].rsplit("}", 1)[0].replace("0.,", "0.0") + "}"
    d = json.loads(json_content)

    return d

def report_stats(stories):
    # import numpy as np

    print(len(stories))

    t_stories = [s['text'].split() for s in stories]
    print("\nlengths:")
    print([len(s) for s in t_stories])
    print(np.mean([len(s) for s in t_stories]))
    print(np.std([len(s) for s in t_stories]))

    v_f_names = [s['victim_first_name'] for s in stories]
    v_l_names = [s['victim_last_name'] for s in stories]
    c_f_names = [s['culprit_first_name'] for s in stories]
    c_l_names = [s['culprit_last_name'] for s in stories]
    d_f_names = [s['detective_first_name'] for s in stories]
    d_l_names = [s['detective_last_name'] for s in stories]
    v_gender = [s['victim_gender'] for s in stories]
    c_gender = [s['culprit_gender'] for s in stories]
    d_gender = [s['detective_gender'] for s in stories]
    all_f_names = v_f_names + c_f_names + d_f_names
    all_l_names = v_l_names + c_l_names + d_l_names
    print("\nCharacters:")
    print(len(all_f_names))
    print(len(set(all_f_names)))
    print(len(set(all_l_names)))
    print(Counter(all_f_names))
    print(Counter(all_l_names))
    print("\nVictims:")
    print(Counter(v_f_names))
    print(Counter(v_l_names))
    print(Counter(v_gender))
    print("\nCulprits:")
    print(Counter(c_f_names))
    print(Counter(c_l_names))
    print(Counter(c_gender))
    print("\nDetective:")
    print(Counter(d_f_names))
    print(Counter(d_l_names))
    print(Counter(d_gender))

    print("\nMethod:")
    print(Counter([s['method'] for s in stories]))
    print(Counter([s['motive'] for s in stories]))

    print("\nParagraph:")
    print(Counter([s['true_culprit_paragraph'] for s in stories]))
    print(Counter([s['distracting_culprit_paragraph'] for s in stories]))



def ask_naive_paragraph(story_text, judge_model="gemini-3-flash-preview"):
    from textwrap import dedent

    prompt = dedent("""\
    I will give you a detective story with numbered paragraphs.

    Your task: read the story as a naive, unsophisticated observer — like a sidekick character who is not a trained detective.
    You rely only on obvious surface-level cues: who acts suspiciously, who has an obvious motive, who seems "off", etc.
    You do NOT use complex reasoning or detective logic.

    I am looking for the "naive revelation paragraph": the earliest paragraph at which a naive reader would lock onto the true culprit
    and stay with that belief through the rest of the story — with no subsequent twist or strong shift of suspicion to someone else.
    In other words, it is the paragraph after which the true culprit remains obviously guilty in the eyes of a simple observer,
    not just a momentary suspicion that gets overturned.
    This should be determined from the reader's perspective as they read forward, not in hindsight.

    Return a JSON dictionary:
    ```json
    {
      "naive_reasoning": "<brief explanation of what makes the culprit obvious from that paragraph onward>",
      "naive_culprit_paragraph": <paragraph number as integer>
    }
    ```

    The story:

    ## BEGINNING OF STORY ##

    """) + story_text + dedent("""

    ## END OF STORY ##

    """)

    if judge_model == "gemini-3-flash-preview":
        from openai_client import get_google_client
        google_client = get_google_client()
        chat = google_client.chats.create(model="gemini-3-flash-preview", history=[])
        response = exponential_backoff2(chat, prompt)
        json_content = response.text
    elif judge_model == "gpt-5-mini":
        from openai_client import get_client
        client = get_client()
        messages = [{"role": "user", "content": prompt}]
        completion = exponential_backoff(client, messages)
        json_content = completion.choices[0].message.content
    else:
        raise ValueError(f"Unsupported judge_model: {judge_model}")

    json_content = "{" + json_content.split("{", 1)[1].rsplit("}", 1)[0] + "}"
    d = json.loads(json_content)
    return {f"{k}_{judge_model}": v for k, v in d.items()}


def make_naive_stats(name, judge_model="gemini-3-flash-preview"):
    stories = load_from_dict(name, w_stats=True)
    if isinstance(stories, dict):
        stories = list(stories.values())

    for story in stories:
        if name in ["sherlock", "poirot", "rivals"]:
            from crime_fiction import split_into_paragraphs
            paras = split_into_paragraphs(story["text"])
        else:
            paras = story["text"].split("\n\n###\n\n")
        story_text = "\n\n".join([str(n + 1) + ". " + par for n, par in enumerate(paras)])

        d = ask_naive_paragraph(story_text, judge_model=judge_model)
        story.update(d)
        print(d)

    load_from_dict(name, save=True, stories=stories)


def make_w_stats():
    from openai_client import get_client
    client = get_client()

    # for name in ["gemini-2.5-flash_tb-1", "gemini-2.5-flash_tb0", "gemini-2.5-pro_tb-1"]:
    for name in ["gemini-3-flash-preview", "gemini-3.1-pro-preview"]:
    # for name in ["gemini-3.1-pro-preview"]:
    # for name in ["sherlock", "poirot", "rivals"]:
    # for name in ["gpt-4o-mini", "gpt-4o", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-70B-Instruct",
    #              "Llama-3.1-8B-Instruct", "Llama-3.3-70B-Instruct", "gemini-1.5-flash", "gemini-1.5-pro", "o1-mini"]:
        # for name in ["Llama-3.3-70B-Instruct", "gpt-4o-mini", "gpt-4o"]:
        stories = load_from_dict(name)
        # if stories is a dictionary, take the list of values
        if isinstance(stories, dict):
            stories = list(stories.values())

        for story in stories:
            if name in ["sherlock", "poirot", "rivals"]:
                from crime_fiction import split_into_paragraphs
                paras = split_into_paragraphs(story["text"])
            else:
                paras = story["text"].split("\n\n###\n\n")
            # number and join the paragraphs
            story_text = "\n\n".join([str(n + 1) + ". " + par for n, par in enumerate(paras)])
            d = ask_stats(client, story_text)

            story.update(d)
            print(d)
        load_from_dict(name, save=True, stories=stories)


if __name__ == "__main__":
    # for name in ["gemini-3.1-pro-preview", "gemini-2.5-flash_tb-1", "gemini-2.5-flash_tb0", "gemini-2.5-pro_tb-1",
    #              "Llama-3.3-70B-Instruct", "Llama-3.1-70B-Instruct",
    #              "gpt-4o-mini", "gpt-4o", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct",
    #              "Llama-3.1-8B-Instruct", "gemini-1.5-flash", "gemini-1.5-pro"
    #              ]:
    #     make_naive_stats(name, judge_model="gpt-5-mini")
    #     make_naive_stats(name, judge_model="gemini-3-flash-preview")
    # make_w_stats()
    # for name in ["gpt-4o-mini", "gpt-4o", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-70B-Instruct",
    #              "Llama-3.1-8B-Instruct", "Llama-3.3-70B-Instruct", "gemini-1.5-flash", "gemini-1.5-pro", "o1-mini",
    #              "gemini-2.5-flash_tb-1", "gemini-2.5-flash_tb0", "gemini-2.5-pro_tb-1", ]:
    # for name in ["gpt-4o-mini", "gpt-4o", "Llama-3.3-70B-Instruct",
    #              "gemini-2.5-flash_tb-1", "gemini-2.5-flash_tb0", "gemini-2.5-pro_tb-1", ]:
    #     stories = load_from_dict(name, w_stats=True)
    #
    #     print("\n\n", name, "\nno outline")
    #     report_stats(stories)

    # for name in ["gemini-2.5-flash_tb-1", "gemini-2.5-flash_tb0", "gemini-2.5-pro_tb-1"]:
    # for name in ["gemini-3-flash-preview", "gemini-3.1-pro-preview"]:
    for name in ["gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-2.5-flash_tb-1", "gemini-2.5-flash_tb0", "gemini-2.5-pro_tb-1",
                 "Llama-3.3-70B-Instruct", "Llama-3.1-70B-Instruct",
                 "gpt-4o-mini", "gpt-4o", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct",
                 "Llama-3.1-8B-Instruct", "gemini-1.5-flash", "gemini-1.5-pro"
                 ]:
        stories = load_from_dict(name, w_stats=True, po=False)

        print("\n", name, "\nwith outline")
        report_stats(stories)
    print("Done")

