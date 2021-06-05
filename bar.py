import json
import subprocess

# with open('bad_chairs2.json') as f:
#     bad_chairs = json.load(f)["data"]

with open('bad_chairs.json') as f:
    chairs = json.load(f)["data"]

num_chairs = len(chairs)
print("num chairs:", num_chairs)

# new_chairs = []
# bad_counter = 0
# for chair_id in chairs:
#     if chair_id in bad_chairs:
#         bad_counter += 1
#     else:
#         new_chairs.append(chair_id)

# print(bad_counter, "bad chairs")

# with open('new_chair.json', 'w') as outfile:
#     json.dump({"3D-FUTURE-model": {"chair": new_chairs}}, outfile)


with open('fix_rotate.json') as f:
    fix_rotate = json.load(f)["data"]
with open('fix_floor.json') as f:
    fix_floor = json.load(f)["data"]
with open('fix_floor_and_rotate.json') as f:
    fix_floor_and_rotate = json.load(f)["data"]


bad = []

i = 0
for chair_id in chairs:
    if chair_id not in fix_rotate:
        continue
    print(i, "of", num_chairs, chair_id)

    chair_path = '../data/3D-FUTURE-model/chair/' + chair_id + '/normalized_model.obj'
    process = subprocess.Popen(['meshlab', chair_path], stdout=subprocess.DEVNULL)
    output, error = process.communicate()

    a = input("bad?")
    if len(a) > 0:
        bad.append(chair_id)

    i += 1

print("bad", bad)

with open('bad_chairsx.json', 'w') as f:
    json.dump({"data": bad}, f)
