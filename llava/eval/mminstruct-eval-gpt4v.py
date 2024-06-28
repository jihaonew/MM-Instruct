import base64
import argparse
from mimetypes import guess_type
import json
import os
import tqdm
import warnings
from openai import AzureOpenAI
import time
NUM_SECONDS_TO_SLEEP = 0.5

rule =  {"role": "Assistant", "prompt": "The above has given a description of a image, \
         instructions given based on this image, and two answers based on the instructions. \
         The following are three evaluation options for comparing the two answers: \
         \nA. Answer A is significantly better. \nB. Answer B is significantly better. \nC.Neither is significantly better.\n \
         Please carefully review both responses and choose one of the options above for your evaluation. \
         Ensure that your selection is indicated by a single letter only."}
api_base = "https://mmdata-openai.openai.azure.com/"
api_key= os.environ.get('OPENAI_API_KEY')
deployment_name = 'gpt4v'
api_version = '2023-12-01-preview'
client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    azure_endpoint = api_base
)

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream' 
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def get_eval(content: str, max_tokens: int, img_path: str):
    while True:
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for evaluating the answer.'
                    }, 
                    {
                    'role': 'user',
                    'content': 
                    [
                        {"type": "image_url","image_url": {"url": local_image_to_data_url(img_path)}},
                        {"type": "text", "text": content },  
                    ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.2
            )
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
    return response.choices[0].message.content.strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument("--image-folder", type=str)
    parser.add_argument('-q', '--question')
    parser.add_argument('-m1', '--model1')
    parser.add_argument('-m2', '--model2')
    args = parser.parse_args()
    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.model1))
    f_ans2 = open(os.path.expanduser(args.model2))
    Tie = M1_win = M2_win = 0
    for ques_js, ans1_js, ans2_js in tqdm.tqdm(zip(f_q, f_ans1, f_ans2)):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)
        cap_str = ques['caption']
        prompt = rule['prompt']
        role = rule['role']
        content1 = (f'[Instruction]\n{ques["text"]}\n\n'
                   f'[Answer A]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[Answer B]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        content2 = (f'[Instruction]\n{ques["text"]}\n\n'
                   f'[Answer A]\n{ans2["text"]}\n\n[End of {role} 1]\n\n'
                   f'[Answer B]\n{ans1["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        scores1 = get_eval(content1, args.max_tokens, os.path.join(args.image_folder, ques['image']))
        scores2 = get_eval(content2, args.max_tokens, os.path.join(args.image_folder, ques['image']))
        if scores1 == scores2: Tie += 1
        elif scores1 == 'A' or scores2 == 'B': M1_win += 1
        elif scores1 == 'B' or scores2 == 'A': M2_win += 1
        else: warnings.warn("Unexpected output from gpt4v! please review defined rule")

    print(f"You Win: {M1_win}, \
          Baseline Win: {M2_win}, \
          Tie: {Tie}")
