import os

processed_dir = os.listdir("../data/processed")

processed_dir_pdbs = [item[:4] for item in processed_dir]


with open("../data/list2.txt", "r") as file:
    items_rem = [line.strip() for line in file]

item_rem_pdbs = [item[:4] for item in items_rem]


## view exists
count = 0
for i in range(len(item_rem_pdbs)):
    if item_rem_pdbs[i] in processed_dir_pdbs:
        count += 1
print("Exists: ", count)

## remove
# count = 0
# for i in range(len(item_rem_pdbs)):
#     if item_rem_pdbs[i] in processed_dir_pdbs:
#         if os.path.exists(f"../data/processed/{item_rem_pdbs[i]}.npz"):
#             os.remove(f"../data/processed/{item_rem_pdbs[i]}.npz")
#             count += 1
# print("Removed count: ", count)