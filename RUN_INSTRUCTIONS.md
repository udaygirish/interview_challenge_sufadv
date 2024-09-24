# RUN Instructions

### For Installing the Dependencies
Install the requirements

    pip3 install -r requirements.txt\
---
Preferred Python Version - 3.8 Above

---
### For Running the  code
Run below command

    python3 main.py 
    args:
    --config : Config Path default set to config/pipeline_config.yaml
    --device : cpu or cuda but default it checks whether device has GPU or CPU and takes the highest priority
    --kvcache : Experimental KV Cache Implementation (True or False)
        With KV Cache parameters the model initialization changes.
---