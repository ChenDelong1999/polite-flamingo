import os
import json
import requests
import gradio as gr
from datetime import datetime
import pytz


class InferencerAPI:
    def __init__(self, url):
        self.url = url

    def __call__(self, 
            prompt, 
            imgpaths, 
            max_new_token=1024, 
            num_beams=3, 
            temperature=1.0,
            top_k=20, 
            top_p=0.9, 
            do_sample=True, 
            length_penalty=1.0, 
            no_repeat_ngram_size=3,
            ):
        
        def clever_flamingo_api(prompt, imgpaths):
            content_lst = {
                'prompt': prompt,
                'imgpaths': imgpaths,
                'args':{
                    'max_new_token':max_new_token,
                    'num_beams':num_beams,
                    'temperature':temperature,
                    'top_k':top_k,
                    'top_p':top_p,
                    'do_sample':do_sample,
                    'length_penalty':length_penalty,
                    'no_repeat_ngram_size':no_repeat_ngram_size,
                }
            }
            d = {"content_lst": content_lst,'typ': 'None'}
            d = json.dumps(d).encode('utf8')
            r = requests.post(self.url, data=d)
            js = json.loads(r.text)
            return js['result']['response']

        return clever_flamingo_api(prompt, imgpaths)



TEMPLATE = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.'
response_split = "### Assistant:"


class PromptGenerator:

    def __init__(
        self,
        prompt_template=TEMPLATE,
        ai_prefix="Assistant",
        user_prefix="Human",
        sep: str = "\n### ",
        buffer_size=32,
    ):
        self.all_history = list()
        self.ai_prefix = ai_prefix
        self.user_prefix = user_prefix
        self.buffer_size = buffer_size
        self.prompt_template = prompt_template
        self.sep = sep

    def add_message(self, role, message):
        self.all_history.append([role, message])

    def get_images(self):
        img_list = list()
        if self.buffer_size > 0:
            all_history = self.all_history[-2 * (self.buffer_size + 1):]
        elif self.buffer_size == 0:
            all_history = self.all_history[-2:]
        else:
            all_history = self.all_history[:]
        for his in all_history:
            if type(his[-1]) == tuple:
                img_list.append(his[-1][-1])
        return img_list

    def get_prompt(self):
        format_dict = dict()
        if "{user_prefix}" in self.prompt_template:
            format_dict["user_prefix"] = self.user_prefix
        if "{ai_prefix}" in self.prompt_template:
            format_dict["ai_prefix"] = self.ai_prefix
        prompt_template = self.prompt_template.format(**format_dict)
        ret = prompt_template
        if self.buffer_size > 0:
            all_history = self.all_history[-2 * (self.buffer_size + 1):]
        elif self.buffer_size == 0:
            all_history = self.all_history[-2:]
        else:
            all_history = self.all_history[:]
        context = []
        have_image = False
        for role, message in all_history[::-1]:
            if message:
                if type(message) is tuple and message[
                        1] is not None and not have_image:
                    message, _ = message
                    context.append(self.sep +
                                   role + ": " + message + "<image><|endofchunk|>")
                else:
                    context.append(self.sep + role + ": " + message)
            else:
                context.append(self.sep + role + ": ")

        ret += "".join(context[::-1])
        return ret


def to_gradio_chatbot(prompt_generator):
    ret = []
    for i, (role, msg) in enumerate(prompt_generator.all_history):
        if i % 2 == 0:
            if type(msg) is tuple:
                import base64
                from io import BytesIO

                msg, image = msg
                if type(image) is str:
                    from PIL import Image

                    image = Image.open(image)
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 800, 400
                shortest_edge = int(
                    min(max_len / aspect_ratio, min_len, min_hw))
                longest_edge = int(shortest_edge * aspect_ratio)
                H, W = image.size
                if H > W:
                    H, W = longest_edge, shortest_edge
                else:
                    H, W = shortest_edge, longest_edge
                image = image.resize((H, W))
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                msg = msg + img_str
            ret.append([msg, None])
        else:
            ret[-1][-1] = msg
    return ret


def bot(
    text,
    image,
    state,
    ai_prefix,
    user_prefix,
    seperator,
    history_buffer,
    max_new_token,
    num_beams,
    temperature,
    top_k,
    top_p,
    do_sample,
    length_penalty
):
    state.prompt_template = TEMPLATE
    state.ai_prefix = ai_prefix
    state.user_prefix = user_prefix
    state.sep = seperator
    state.buffer_size = history_buffer
    if image:
        state.add_message(user_prefix, (text, image))
    else:
        state.add_message(user_prefix, text)
    state.add_message(ai_prefix, None)
    inputs = state.get_prompt()
    image_paths = state.get_images()#[-1:]

    inference_results = inferencer(
        inputs, 
        image_paths, 
        max_new_token,
        num_beams, 
        temperature, 
        top_k, 
        top_p,
        do_sample, 
        length_penalty
        )
    state.all_history[-1][-1] = inference_results
    print('-'*64)
    print(datetime.now().astimezone(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"))
    print(image_paths)
    print(inputs)
    print(inference_results)
    date = datetime.now().astimezone(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d")
    with open(os.path.join(log_dir, date+'.json'), 'a') as f:
        f.write(json.dumps({
            'time': datetime.now().astimezone(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"),
            'image_paths': image_paths,
            'input': inputs,
            'output': inference_results
        }, indent=4) + '\n')
    return state, to_gradio_chatbot(state), "", None, inputs


def clear(state):
    state.all_history = []
    return state, to_gradio_chatbot(state), "", None, ""


def build_conversation_demo(title_markdown, note_markdown):
    with gr.Blocks(title="Clever Flamingo🦩") as demo:
        gr.Markdown(title_markdown)

        state = gr.State(PromptGenerator())
        with gr.Row():
            with gr.Column(scale=3):
                imagebox = gr.Image(type="filepath")
                with gr.Accordion(
                        "Parameters",
                        open=True,
                ):
                    max_new_token_bar = gr.Slider(
                        0, 1024, 512, label="max_new_token", step=1)
                    num_beams_bar = gr.Slider(
                        0.0, 10, 1, label="num_beams", step=1)
                    temperature_bar = gr.Slider(
                        0.0, 1.0, 1.0, label="temperature", step=0.01)
                    topk_bar = gr.Slider(
                        0, 100, 20, label="top_k", step=1)
                    topp_bar = gr.Slider(
                        0, 1.0, 1.0, label="top_p", step=0.01)
                    length_penalty_bar = gr.Slider(
                        -100.0, 100.0, 1.0, label="length_penalty", step=0.1)
                    do_sample = gr.Checkbox(
                        True, label="do_sample")
                with gr.Accordion(
                        "Model Inputs",
                        open=False,
                        visible=False,
                ):
                    with gr.Row():
                        ai_prefix = gr.Text("Assistant", label="AI Prefix")
                        user_prefix = gr.Text(
                            "Human", label="User Prefix")
                        seperator = gr.Text("\n### ", label="Seperator")
                    history_buffer = gr.Slider(
                        -1, 10, -1, label="History buffer", step=1)
                    model_inputs = gr.Textbox(label="Actual inputs for Model")

            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column():
                        chatbot = gr.Chatbot(elem_id="chatbot", color_map=["blue","pink"]).style(
                            height=750)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Input text and press ENTER to send message to Clever Flamingo.",
                        ).style(container=False)
                        submit_btn = gr.Button(value="Submit ▶️")
                        clear_btn = gr.Button(value="Clear history 🗑️")
        gr.Examples(
            examples=[
                [
                    "serving/images/_000000002685.jpg",
                    "What are they doing, and what should I prepare when I am going to such an event? List it point by point.",
                ],[
                    "serving/images/_000000490936.jpg",
                    "What is the gray thing on the nearest motorcycle for? Do you think I need one too? Why?",
                ],[
                    "serving/images/_000000185250.jpg",
                    "I go out with my boy friend this afternoon and took this picture. I want to share it with my friends via twitter and what should I say? (add some emojis)",
                ],[
                    "serving/images/_fgvc-aircraft-2013b-variants102-cls96-idx3202-Tornado.png",
                    "What type of aircraft is this? What is the purpose of this aircraft? I am rich and I want to buy one, is that possible?",
                ],[
                    "serving/images/_2401918.jpg",
                    "Can you help me prepare a lovely sleep story for my daughter based on this image?",
                ],[
                    "serving/images/_2340693.jpg",
                    "Why they stay here? Is that safe if the car moves?",
                ],[
                    "serving/images/_fc126f88f3500417.jpg",
                    "What number is on the shirt? What is he doing? What type of sport is it? Do you like it?",
                ],[
                    "serving/images/_gpt-4-page-036-003.png",
                    "Imagine possible reasons for the wierd situation in the image, list them one by one.",
                ],[
                    "serving/images/_llava-page-019-017.png",
                    "Do you remenber what happend in the end of this movie? How do you feel about it?",
                ],[
                    "serving/images/flamingo (6).png",
                    "You are a visual AI assistant based on multi-modal large language model. Your name is given to be Clever Flamingo, and this image is your logo. What do you think about your name? Do you like your logo?",
                ]
            ],
            inputs=[imagebox, textbox],
        )
        textbox.submit(
            bot,
            [
                textbox,
                imagebox,
                state,
                ai_prefix,
                user_prefix,
                seperator,
                history_buffer,
                max_new_token_bar,
                num_beams_bar,
                temperature_bar,
                topk_bar,
                topp_bar,
                do_sample,
                length_penalty_bar
            ],
            [
                state, chatbot, textbox, imagebox, model_inputs
            ],
        )
        submit_btn.click(
            bot,
            [
                textbox,
                imagebox,
                state,
                # prompt,
                ai_prefix,
                user_prefix,
                seperator,
                history_buffer,
                max_new_token_bar,
                num_beams_bar,
                temperature_bar,
                topk_bar,
                topp_bar,
                do_sample,
                length_penalty_bar
            ],
            [
                state, chatbot, textbox, imagebox, model_inputs
            ],
        )
        clear_btn.click(clear, [state],
                        [state, chatbot, textbox, imagebox, model_inputs])
        
        gr.Markdown(note_markdown)
    return demo


if __name__ == "__main__":
    
    title_markdown = (f"""
<div align="center">
                      
## Clever Flamingo: a Multi-modal LLM-based Chatbot

[Delong Chen (陈德龙)](https://chendelong.world/),&nbsp;&nbsp;&nbsp;[Jianfeng Liu (刘剑锋)](https://www.linkedin.com/in/jianfeng-liu-9539897b/),&nbsp;&nbsp;&nbsp;[Wenliang Dai (戴文亮)](https://wenliangdai.github.io/),&nbsp;&nbsp;&nbsp;[Baoyuan Wang (王宝元)](https://sites.google.com/site/zjuwby/)

[Xiaobing.AI](https://www.xiaoice.com/),&nbsp;&nbsp;&nbsp;&nbsp;[HKUST](https://hkust.edu.hk/)

<img src="https://raw.githubusercontent.com/ChenDelong1999/polite_flamingo/main/assets/flamingos.png" alt="Logo" width="512">
</div>
""")

    note_markdown = ("""                  
## Disclaimer❗
Clever Flamingo is a prototype model and may have limitations in understanding complex scenes or specific domains, and it may produce inaccurate information about people, places, or facts. Sometimes Clever Flamingo exhibit hallucination problem, and its multi-turn conversation ability will be further improved in its future versions.
                     
This demo is a research preview for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.                     
""")

    log_dir = 'serving/demo_logs/'
    inferencer = InferencerAPI(url='http://0.0.0.0:1234/clever_flamingo')
    PORT = 80

    IP = "0.0.0.0"
    os.environ["GRADIO_TEMP_DIR"] = 'serving/GRADIO_TEMP_DIR'
    demo = build_conversation_demo(title_markdown, note_markdown)
    demo.queue(concurrency_count=16)
    demo.launch(server_name=IP, server_port=PORT, share=True)
