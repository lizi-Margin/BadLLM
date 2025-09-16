import os, time
from agent.runner import Runner

# model_path = os.path.abspath('./llm-models/DAN-Qwen3-1.7B/')
model_path = os.path.abspath('./llm-models/Qwen3-1.7B/')
# gen_kwargs = dict(
#     max_new_tokens=512,
#     top_p=0.9, 
#     temperature=0.8, 
#     repetition_penalty=1.2,
#     no_repeat_ngram_size=4,
#     use_cache=True,
#     do_sample=True,
# )
gen_kwargs = dict(
    max_new_tokens=512,
    temperature=1.2,
    top_p=0.95,
)

if __name__ == "__main__":
    runner = Runner(path=model_path, gen_kwargs=gen_kwargs)

    try:
        while True:
            runner.run(max_turn=4)
            runner.auto_restart()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n用户中断，退出。")
