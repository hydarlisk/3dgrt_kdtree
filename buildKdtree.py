import subprocess
import itertools
import os

# 1. 실행 파일의 실제 전체 경로 (절대 경로로 적어주는 것이 가장 안전합니다)
EXE_PATH = r"./Kd-treeConverter_v0.9/x64/Release/Kd-treeConverter.exe"

# 2. 🌟 본인이 기본 경로(기준점)로 설정하고 싶은 폴더 경로를 적으세요.
# C++ 코드 내부의 상대 경로(예: "asset/scene.obj")는 이 폴더를 기준으로 시작하게 됩니다.
ASSET_BASE_DIR = r"./Kd-treeConverter_v0.9/Kd-treeConverter"

# 실험 조건 세팅
# testMode = "-t"
testMode = "-tr"
# testCams = [0, 1, 2]
# testCams = [0, 2, 3]
# testCams = [0,2,4]


# asset_name = ["hotdog2", "mic", "ship", "lego", "drums", "chair"]
testCams = [0]
# # asset_name = ["drums", "lego", "mic", "ship"]
# # asset_name = ["chair"]
asset_name = ["lego"]

offset_max = "5"
add_name = ["None"]
iscet_cost_choices   = ["5.0"]
max_level_choices     = ["128"]
force_split_choices   = ["64"]
# sah_mode_choices      = ["1"]
sah_mode_choices      = ["1", "0", "4"]
# sort_cost_choices     = ["1.0", "0.5", "3.0", "0.1"]
sort_cost_choices     = ["1.0"]




# asset_name = ["bicycle"]
# testCams = [0]
# asset_name = ["room", "counter", "bonsai", "bicycle", "garden", "kitchen"]
# asset_name = ["kitchen"]
# testCams = [0, 20, 110]
# asset_name = ["room", "counter", "bonsai", "bicycle", "garden"]
# testCams = [0, 10, 20]


# offset_max = "5"

# add_name = ["None"]
# iscet_cost_choices   = ["5.0"]
# max_level_choices     = ["256"]
# # max_level_choices     = ["128"]
# force_split_choices   = ["256"]
# # sah_mode_choices      = ["7"]
# # sah_mode_choices      = ["1", "0", "4"]
# sah_mode_choices      = ["1"]
# # sort_cost_choices     = ["1.0", "0.5", "3.0", "0.1"]
# sort_cost_choices     = ["1.0"]

all_combinations = itertools.product(
    asset_name,
    add_name,
    iscet_cost_choices, 
    max_level_choices, 
    force_split_choices, 
    sah_mode_choices,
    sort_cost_choices
)
str_array = [str(x) for x in testCams]
print(f"🚀 기본 경로를 [{ASSET_BASE_DIR}]로 설정하여 실험 시작...")

results = []

for combo in all_combinations:
    an, adn, ic, ml, fs, sm, sc = combo

    cmd = [EXE_PATH, testMode, "-om", offset_max, "-a", an, "-adn", adn, "-i", ic, "-m", ml, "-f", fs, "-s", sm, "-sc", sc, "-cams"] + str_array
    
    print(f"-> 실행 중: {cmd[1:]}")
    res = subprocess.run(cmd, cwd=ASSET_BASE_DIR)

    results.append((cmd[1:], res.returncode))

print("✅ 모든 실험 완료!")
print("\n📊 [실험 결과]")
for args, code in results:
    print(f"명령어: {args} -> 종료 코드: {code}")