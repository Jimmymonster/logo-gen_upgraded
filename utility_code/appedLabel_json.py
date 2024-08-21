import requests
import json

# Configuration
def load_env_file(filepath):
    env_vars = {}
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    return env_vars

label_studio_project_id = 218

output_json_path = "append_annotation.json"
class_name = "TNN"
image_size = (704,576)
bounding_box = (617, 87, 670, 177) 
env_vars = load_env_file('.env')

label_studio_ip = env_vars['LABEL_STUDIO_IP']
api_key = env_vars['API_KEY'] 



# API call to fetch the JSON data
api_call = f"http://{label_studio_ip}/api/projects/{label_studio_project_id}/export?exportType=JSON&download_all_tasks=true"
headers = {
    "Authorization": f"Token {api_key}"
}

# Fetch JSON data from Label Studio
response = requests.get(api_call, headers=headers)
response.raise_for_status()  # Check if the request was successful
data = response.json()

# Add bounding box to each annotation
for task in data:
    if isinstance(task.get('annotations'), list):
    # Example assumes annotations are in the 'annotations' field
        for annotation in task['annotations']:
            # Add bounding box to the annotation
            annotation['result'].append({
                "type": "rectanglelabels",
                "value": {
                    "x": bounding_box[0]/image_size[0]*100,
                    "y": bounding_box[1]/image_size[1]*100,
                    "width": (bounding_box[2] - bounding_box[0])/image_size[0]*100,
                    "height": (bounding_box[3] - bounding_box[1])/image_size[1]*100,
                    "rectanglelabels": [class_name]
                },
                "from_name": "label",
                "to_name": "image",
                "origin": "manual",
            })

# Save updated JSON data to a file
with open(output_json_path, 'w') as outfile:
    json.dump(data, outfile, indent=4)

print(f"Updated JSON file saved to {output_json_path}")
