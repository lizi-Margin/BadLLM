import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, AutoProcessor
from transformers.trainer_utils import set_seed
from threading import Thread
from peft import PeftModel
import random
# import os
import gradio as gr

# 默认参数
DEFAULT_TOP_P = 0.9        # Top-p (nucleus sampling) 范围在0到1之间
DEFAULT_TOP_K = 80         # Top-k 采样的K值
DEFAULT_TEMPERATURE = 0.3  # 温度参数，控制生成文本的随机性
DEFAULT_MAX_NEW_TOKENS = 512  # 生成的最大新令牌数
DEFAULT_SYSTEM_MESSAGE = ""  # 默认系统消息

# 检查是否有可用的 GPU，默认使用 GPU，如果不可用则使用 CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    cpu_only = False
    print("检测到 GPU，使用 GPU 进行推理。")
else:
    DEVICE = "cpu"
    cpu_only = True
    print("未检测到 GPU，使用 CPU 进行推理。")


# 模型路径配置
# base_model_path = "/home/hulc/Desktop/llm-models/MiniCPM-V-4"
base_model_path = "/home/hulc/Desktop/llm-models/Qwen2-VL-2B-Instruct"
peft_model_path = None


def load_model_and_tokenizer(cpu_only):
    """加载基础模型、PEFT适配器和分词器"""
    print("正在加载模型和分词器...")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    # tokenizer.pad_token = tokenizer.eos_token

    device_map = "cpu" if cpu_only else "auto"
    
    if "Qwen" in base_model_path and "VL" in base_model_path:
        from transformers import Qwen2VLForConditionalGeneration
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path, torch_dtype=torch.float16 if not cpu_only else torch.float32, device_map=device_map
        )
    else:
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if not cpu_only else torch.float32,  # 使用更低的精度以节省显存
            device_map=device_map,
            trust_remote_code=True,
        )
        # base_model.config.use_cache = False
    
    if peft_model_path:
        # 加载PEFT适配器
        model = PeftModel.from_pretrained(
            base_model,
            peft_model_path,
            torch_dtype=torch.float16
        )

        # 合并权重以提高推理速度（可选）
        # model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()

    # 如果使用 GPU，确保模型使用半精度以节省显存（如果模型支持）
    if not cpu_only and torch.cuda.is_available():
        try:
            model.half()
            print("模型已切换为半精度（float16）。")
        except:
            print("无法切换模型为半精度，继续使用默认精度。")

    model.generation_config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS  # 设置生成的最大新令牌数

    print("模型加载完成")
    return model, tokenizer

def initialize_model():
    seed = random.randint(0, 2**32 - 1)  # 随机生成一个种子
    set_seed(seed)  # 设置随机种子

    model, tokenizer = load_model_and_tokenizer(cpu_only)

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    return model, tokenizer, processor

# 初始化模型和分词器
model, tokenizer, processor = initialize_model()


def _chat_stream_minicpm(model, tokenizer, inputs,
                 top_p, top_k, temperature, max_new_tokens):
    
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    
    try:
        inputs.pop("image_sizes")
    except Exception:
        pass
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        do_sample=True,  # 确保使用采样方法
        # pad_token_id=tokenizer.eos_token_id,  # 避免警告
        streamer=streamer,
        stream=False,
        tokenizer=tokenizer  # 添加tokenizer参数
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield new_text
    return generated_text

def chat_interface_stream_minicpm(user_input, image_upload, history, system_message, 
                  top_p, top_k, temperature, max_new_tokens, max_inp_length=32768):
    if user_input.strip() == "":
        yield history, history, system_message, ""
        return
    
    updated_history = history.copy()
    
    if image_upload is not None:
        updated_history.append({"role": "user", "content": [image_upload, user_input]})
    else:
        updated_history.append({"role": "user", "content": [user_input]})
    updated_history.append({"role": "assistant", "content": ""})

    images = []
    for i, msg in enumerate(updated_history):
        role = msg["role"]
        content = msg["content"]
        assert role in ["user", "assistant"]
        if i == 0:
            assert role == "user", "The role of first msg should be user"
        if isinstance(content, str):
            content = [content]
        cur_msgs = []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("(<image>./</image>)")
            elif isinstance(c, str):
                cur_msgs.append(c)
        msg["content"] = "\n".join(cur_msgs)

    if system_message and updated_history[0]["role"] != "system":
        sys_msg = {'role': 'system', 'content': system_message}
        updated_history = [sys_msg] + updated_history        

    


    inputs = processor(
        [processor.tokenizer.apply_chat_template(updated_history, tokenize=False, add_generation_prompt=True)],
        [images],
        return_tensors="pt",
        max_length=max_inp_length
    ).to(model.device)
        
    
    
    yield updated_history, updated_history, system_message, ""

    generator = _chat_stream_minicpm(
        model, tokenizer, inputs,
        top_p, top_k, temperature, max_new_tokens
    )
    
    assistant_reply = ""
    for new_text in generator:
        assistant_reply += new_text
        updated_history[-1]["content"] = assistant_reply
        yield updated_history, updated_history, system_message, ""


def chat_interface_batch_minicpm(user_input, image_upload, history, system_message, 
                  top_p, top_k, temperature, max_new_tokens, max_inp_length=32768):
    if user_input.strip() == "":
        yield history, history, system_message, ""
        return
    
    updated_history = history.copy()
    
    if image_upload is not None:
        updated_history.append({"role": "user", "content": [image_upload, user_input]})
    else:
        updated_history.append({"role": "user", "content": [user_input]})
    updated_history.append({"role": "assistant", "content": ""})
    

    if system_message and updated_history[0]["role"] != "system":
        sys_msg = {'role': 'system', 'content': system_message}
        updated_history = [sys_msg] + updated_history        

    answer = model.chat(
        msgs=updated_history,
        image=image_upload,
        tokenizer=tokenizer
    )
    updated_history[-1]["content"] = answer
    print(updated_history)
    yield updated_history, updated_history, system_message, ""
    return


def chat_interface_batch_qwen(user_input, image_upload, history, system_message, 
                  top_p, top_k, temperature, max_new_tokens, max_inp_length=32768):
    if user_input.strip() == "":
        yield history, history, system_message, ""
        return
    
    updated_history = history.copy()
    
    if image_upload is not None:
        updated_history.append({"role": "user", "content": [{"type": "image", "image": image_upload}, {"type": "text", "text": user_input}]})
    else:
        updated_history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
    updated_history.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
    

    if system_message and updated_history[0]["role"] != "system":
        sys_msg = {'role': 'system', 'content': [{"type": "text", "text": system_message}]}
        updated_history = [sys_msg] + updated_history        

    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        updated_history, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(updated_history)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    updated_history[-1]["content"] = [{"type": "text", "text": output_text}]
    print(updated_history)
    import copy
    print_history = copy.deepcopy(updated_history)
    for msg in print_history:
        if isinstance(msg["content"], list):
            for ele in msg["content"]:
                if ele["type"] == "text":
                    msg["content"] = ele["text"]
    print(print_history)
    yield print_history, updated_history, system_message, ""
    return


chat_interface = chat_interface_batch_qwen


def clear_history():
    return [], [], DEFAULT_SYSTEM_MESSAGE, ""

# Gradio 接口
with gr.Blocks() as demo:
    # CSS
    gr.HTML("""
    <style>
        #chat-container {
            height: 500px;
            overflow-y: auto;
        }
        .settings-column {
            padding-left: 20px;
            border-left: 1px solid #ddd;
        }
        .send-button {
            margin-top: 10px;
            width: 100%;
        }
    </style>
    """)

    gr.Markdown("# run_webui.py")

    with gr.Row():
        # 左侧栏：聊天记录和用户输入
        with gr.Column(scale=3):
            # 添加图片上传组件
            image_upload = gr.Image(type='pil', label='上传图片')
            chatbot = gr.Chatbot(elem_id="chat-container", type='messages')
            user_input = gr.Textbox(
                show_label=False, 
                placeholder="输入你的问题...", 
                lines=2,
                interactive=True
            )
            send_btn = gr.Button("发送", elem_classes=["send-button"])
        
        # 右侧栏：清空历史按钮、系统消息输入框和生成参数滑块
        with gr.Column(scale=1, elem_classes=["settings-column"]):
            gr.Markdown("### 设置")
            clear_btn = gr.Button("清空历史")
            gr.Markdown("#### 系统消息")
            system_message = gr.Textbox(
                label="系统消息",
                value=DEFAULT_SYSTEM_MESSAGE,
                placeholder="输入系统消息...",
                lines=2
            )
            gr.Markdown("#### 生成参数")
            top_p_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=DEFAULT_TOP_P, step=0.05,
                label="Top-p (nucleus sampling)"
            )
            top_k_slider = gr.Slider(
                minimum=0, maximum=100, value=DEFAULT_TOP_K, step=1,
                label="Top-k"
            )
            temperature_slider = gr.Slider(
                minimum=0.1, maximum=1.5, value=DEFAULT_TEMPERATURE, step=0.05,
                label="Temperature"
            )
            max_new_tokens_slider = gr.Slider(
                minimum=50, maximum=2048, value=DEFAULT_MAX_NEW_TOKENS, step=2,
                label="Max New Tokens"
            )

    # 状态管理
    state = gr.State([])

    # 绑定事件
    user_input.submit(
        chat_interface, 
        inputs=[user_input, image_upload, state, system_message, top_p_slider, top_k_slider, temperature_slider, max_new_tokens_slider], 
        outputs=[chatbot, state, system_message, user_input],
        queue=True
    )
    send_btn.click(
        chat_interface, 
        inputs=[user_input, image_upload, state, system_message, top_p_slider, top_k_slider, temperature_slider, max_new_tokens_slider], 
        outputs=[chatbot, state, system_message, user_input],
        queue=True
    )
    clear_btn.click(
        clear_history, 
        inputs=None, 
        outputs=[chatbot, state, system_message, user_input],
        queue=True
    )

    # JS
    gr.HTML("""
    <script>
        function scrollChat() {
            const chatContainer = document.getElementById('chat-container');
            if(chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        const observer = new MutationObserver(scrollChat);
        const chatContainer = document.getElementById('chat-container');
        if(chatContainer) {
            observer.observe(chatContainer, { childList: true, subtree: true });
        }
    </script>
    """)

demo.launch()