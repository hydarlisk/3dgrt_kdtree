import subprocess
import itertools
import os

# 1. 실행 파일의 실제 전체 경로 (절대 경로로 적어주는 것이 가장 안전합니다)
EXE_PATH = r"./Kd-treeConverter_v0.9/x64/Release/Kd-treeConverter.exe"

# 2. 🌟 본인이 기본 경로(기준점)로 설정하고 싶은 폴더 경로를 적으세요.
# C++ 코드 내부의 상대 경로(예: "asset/scene.obj")는 이 폴더를 기준으로 시작하게 됩니다.
ASSET_BASE_DIR = r"./Kd-treeConverter_v0.9/Kd-treeConverter"

# 실험 조건 세팅
iscet_cost_choices   = ["5.0", "10.0"]
max_level_choices     = ["128"]
force_split_choices   = ["64"]
sah_mode_choices      = ["0", "4"]

all_combinations = itertools.product(
    iscet_cost_choices, 
    max_level_choices, 
    force_split_choices, 
    sah_mode_choices
)

print(f"🚀 기본 경로를 [{ASSET_BASE_DIR}]로 설정하여 실험 시작...")

for combo in all_combinations:
    ic, ml, fs, sm = combo
    
    # 🌟 cwd를 바꿀 때는 실행 파일의 전체 경로(EXE_PATH)를 다 넣어주어야 합니다.
    cmd = [EXE_PATH, "-i", ic, "-m", ml]
    
    print(f"-> 실행 중: {cmd[1:]}")
    subprocess.run(cmd, cwd=ASSET_BASE_DIR)

print("✅ 모든 실험 완료!")