# How to run

## Additional Files to add:
Add a file called API_KEYS.toml
In that add a variable called GOOGLE_API_KEY and set your gemini api key there
Get the API Key here: https://aistudio.google.com/app/apikey

## Modify other settings:
You can additionally modify settings of models before running them in the two config files: ./config.toml and ./VectorDB/config.toml
They contain various model settings like temperature, model-name, device (cpu or gpu) ...etc
Make sure you set every device to run on cpu if you don't have cuda supported gpu
If you indeed have a gpu that supports cuda, then install cuda from here: https://developer.nvidia.com/cuda-toolkit

## Install dependencies:
```
uv sync
```
## Start vector database server:
First open a new terminal
```
cd VectorDB/
uv run db.py
```
## Run the UI:
In another terminal
```
uv run streamlit run main.py
```

# Known issues and how to fix them

## Rate-limit exceeded errors:
Occurs when you are sending too many LLM-API requests in a short span of time or exceeded your daily quota limit
For rate limit information gemini: You can refer here: https://ai.google.dev/gemini-api/docs/rate-limits
A temporary fix to use some other gemini-api key linked with another google-key

## Device not found:
Occurs when you don't have GPU with cuda support. Change the config.toml device settings to cpu mode

## Memory limit exceeded:
This likely occurs when you don't have sufficient memory. It is recommended that you run the vector database embedder (./VectorDB/config.toml) on GPU (set the device under embedder.model_kwargs to cuda) 
and set the device setting under FCoref in ./config.toml to cpu. It is recommended that you have >12 GB of ram/vram when running FCoref for this particular example

# Team Members:
Doddapaneni Udith: https://github.com/udithdoddapaneni
Yash Yadav: https://github.com/DragoCodes
Sanjiv: https://github.com/Sanjiv4321

# Contact Information:
udithdoddapaneni@gmail.comc
sk.sanjiv1234@gmail.com
yashyadavduos52@gmail.com
