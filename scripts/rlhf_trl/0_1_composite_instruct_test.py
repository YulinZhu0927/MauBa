import json
import time

import airsim

input_file = "composite_instruction.jsonl"
output_file = "validated_composite_instruction.jsonl"

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

validated_tasks = []

# Read tasks
with open(input_file, "r", encoding="utf-8") as f:
    tasks = [json.loads(line) for line in f]

for task in tasks:
    print(f"\nTask Description: {task['description']}\n")

    # Execute the generated code
    exec_globals = {"airsim": airsim, "client": client}
    try:
        exec(task["code"], exec_globals)
    except Exception as e:
        print(f"Error executing task: {e}")
        satisfaction = 0
    else:
        # User feedback
        satisfaction = int(input("Enter 0 (unsatisfied) or 1 (satisfied): "))

    time.sleep(1)
    client.reset()
    time.sleep(2)

    # Save feedback to task data
    task["satisfaction"] = satisfaction
    validated_tasks.append(task)

# Save validated tasks
with open(output_file, "w", encoding="utf-8") as f:
    for task in validated_tasks:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")

print("Validation completed and saved.")
