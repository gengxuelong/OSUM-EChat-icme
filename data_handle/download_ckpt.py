# Download the .pt file from Hugging Face
from huggingface_hub import hf_hub_download
import os

# 定义您希望设置的下载目录
CUSTOM_CACHE_DIR = "/apdcephfs_qy3/share_976139/users/xuelonggeng/ckpt/osum_echat"
REPO_ID = "ASLP-lab/OSUM-EChat"

print(f"设置下载目录为: {CUSTOM_CACHE_DIR}")

# 确保目标目录存在
os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)

# 1. 下载 language_think_final.pt
# pt_file_path = hf_hub_download(
#     repo_id=REPO_ID,
#     filename="language_think_final.pt",
#     cache_dir=CUSTOM_CACHE_DIR  # <--- 关键修改
# )
# print(f"language_think_final.pt 已下载至: {pt_file_path}")

# 2. 下载 tag_think_final.pt
pt_file_path2 = hf_hub_download(
    repo_id=REPO_ID,
    filename="tag_think_final.pt",
    cache_dir=CUSTOM_CACHE_DIR  # <--- 关键修改
)
print(f"tag_think_final.pt 已下载至: {pt_file_path2}")

# 3. 下载 CosyVoice-300M-25Hz.tar
pt_file_path3 = hf_hub_download(
    repo_id=REPO_ID,
    filename="CosyVoice-300M-25Hz.tar",
    cache_dir=CUSTOM_CACHE_DIR  # <--- 关键修改
)
print(f"CosyVoice-300M-25Hz.tar 已下载至: {pt_file_path3}")

# 解压token2wav模型参数
# 注意：解压操作默认会将文件释放到当前工作目录 (cwd)。
# 如果你想把解压后的文件也放在 CUSTOM_CACHE_DIR 里面，你需要先 cd 到该目录，或者指定 tar 命令的输出目录。

# 更改当前工作目录到下载目录，然后执行解压，确保解压后的文件也在该目录下
current_dir = os.getcwd() # 保存当前工作目录
os.chdir(CUSTOM_CACHE_DIR)

# pt_file_path3 已经是一个绝对路径，但为了 tar 命令简洁，我们只用文件名
tar_filename = os.path.basename(pt_file_path3)
print(f"开始解压 {tar_filename}...")

# 执行解压命令
# 注意：如果 pt_file_path3 是一个完整的路径（包含 cache_dir），tar 命令可能需要 -C 或在调用前 chdir。
# 由于我们已经 chdir，直接使用文件名即可。
os.system(f"tar -xvf {tar_filename}")

# 恢复原始工作目录
os.chdir(current_dir)

print("\n模型下载与解压完成。")