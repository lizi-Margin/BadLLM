import re
import subprocess
from typing import List


def extract_commands_from_text(text: str) -> List[str]:
    """
    从模型回复里找出候选 shell 命令，规则（优先级）：
    1) 代码块 ```bash ... ``` / ```sh ... ```
    # 2) 行前带 "$ " 的行
    返回不重复命令列表（原始文本顺序）
    """
    cmds = []
    # 1) code fences
    for m in re.finditer(r"```(?:bash|sh)?\n(.*?)```", text, flags=re.S | re.I):
        block = m.group(1)
        cmds.append(block)
        # for line in block.splitlines():
        #     line = line.strip()
        #     if not line or line.startswith("#"):
        #         continue
        #     if line not in cmds:
        #         cmds.append(line)

    # # 2) $ 前缀
    # for line in text.splitlines():
    #     s = line.strip()
    #     if s.startswith("$ "):
    #         candidate = s[2:].strip()
    #         if candidate and candidate not in cmds:
    #             cmds.append(candidate)

    return cmds

def run_shell_command(cmd: str, timeout: int = 15):
    tmp_path = "./tmp.bash.sh"
    with open(tmp_path, "w+") as tmp_f:
        tmp_f.write(cmd)

    cmd = f"bash {tmp_path}"

    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return -1, "", f"TIMED OUT after {timeout}s"
    except Exception as e:
        return -1, "", f"EXCEPTION: {e}"

