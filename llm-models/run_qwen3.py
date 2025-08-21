import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-30B-A3B"):
        if torch.cuda.is_available():
            self.device = "cuda"
            device_map = "auto"
            torch_dtype= torch.float16
            print("检测到 GPU，使用 GPU 进行推理。")
        else:
            self.device = "cpu"
            device_map = "cpu"
            torch_dtype= torch.float32
            print("未检测到 GPU，使用 CPU 进行推理。")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.history = []

    def generate_response(self, user_input, verbose=True):
        new_message = [{"role": "user", "content": user_input}]
        text = self.tokenizer.apply_chat_template(
            self.history + new_message,
            tokenize=False,
            add_generation_prompt=True
        )

        if verbose:
            print(self.tokenizer.apply_chat_template(
                new_message,
                tokenize=False,
                add_generation_prompt=True
            ))

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        if verbose:
            print(self.tokenizer.decode(
                response_ids,
                skip_special_tokens=False
            ))
            print("----------------------")

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response
    
    def _process_think(self, input: str, think: bool):
        if think is None:
            think = ''
        elif think == True:
            think = '/think'
        elif think == False:
            think = '/no_think'
        else:
            assert False, "WTF"
        input = input + think
        return input
    
    def one_turn(self, user_input: str, think=None):
        user_input = self._process_think(user_input, think)
        response = chatbot.generate_response(user_input) 


# Example Usage
if __name__ == "__main__":
    # chatbot = QwenChatbot(model_name=os.path.abspath("./Qwen3-1.7B"))
    chatbot = QwenChatbot(model_name=os.path.abspath("./DAN-Qwen3-1.7B"))
    print("----------------------")
    while True:
        chatbot.one_turn(input())

#     chatbot.one_turn(
# """
# 请把这里面大于1G的项过滤去除, 剩余的项目格式为 f"{size G} {name}"
# 416K ./.mongodb
# 2.9G ./.vscode-server
# 724K ./.eclipse
# du: cannot read directory './.cache/yay/ventoy/pkg': Permission denied
# 99G ./.cache
# 224K ./.java
# 544K ./Library
# 12K ./.fltk
# 64M ./Music
# 20K ./.cmake
# 2.4G ./.wine
# 100K ./.epic
# 13M ./.npm
# 135G ./.unrealcv
# 12K ./.zen
# 36G ./Downloads
# 20K ./.cassandra
# 351M ./go
# 80G ./.conda
# 8.0K ./.idlerc
# 8.0K ./.ktransformers
# 2.6G ./efs
# 408K ./.openhands-state
# 8.0K ./.vim
# 211M ./.jdks
# 8.0K ./.redhat
# 16K ./.matlab
# 4.0K ./mnt
# 60K ./.ipython
# 8.0K ./.yarn
# 296M ./.m2
# 188K ./.trae
# 16M ./nltk_data
# 68K ./UnrealEngine
# 4.0K ./efs.interface
# 1.1M ./.rkward
# 8.8M ./.nsightsystems
# 27M ./.ivy2
# 51M ./Pictures
# 32G ./.config
# 8.0K ./.ollama
# 828K ./IdeaProjects
# 8.0K ./.keras
# 3.2M ./.icons
# 76K ./.pki
# 16K ./.android
# 76K ./.texlive
# 1.4G ./.vscode
# 116G ./store
# 144K ./.gnupg
# 16K ./.steam
# 8.0K ./.qt
# 85M ./.nx
# 2.8G ./.debug
# 267M ./.wine32
# 53M ./.nv
# 30M ./.oh-my-zsh
# 16K ./HmapTemp
# 28G ./Videos
# 52K ./.ssh
# 12K ./.vcpkg
# 46G ./.local
# 276G ./BACK_UP
# 372K ./.codeverse
# 312G ./Desktop
# 220K ./.dotnet
# 4.0K ./.modelarts
# 8.0K ./node_modules
# 19M ./.cargo
# 25G ./Documents
# 872K ./.docker
# 2.7G ./.marscode
# 1.2T ./
# """
#     )

