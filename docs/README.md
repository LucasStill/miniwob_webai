# Navigating WebAI: Training Autonomous Agents to Navigate the Web

This repository contains the code for my master thesis titled "Navigating WebAI: Training Autonomous Agents to Navigate the Web". The project focuses on training autonomous agents to navigate the web using AI techniques. Below is a brief description of some of the files in this repository:

## Files:

### VMPO_remote_gpu.py
- Description: This file contains the implementation of the VMPO algorithm for training autonomous agents on remote GPUs. It must be deployed on a remote GPU with an open connection port. It is the version taking output embeddings.

### VMPO_remote_gpu_linear.py
- Description: Similar to "VMPO_remote_gpu.py", this file contains an alternative implementation of the VMPO algorithm using the final used version of our output that uses linear activations instead of indexing tokens against an embedding table.

## The following implements and test different features during our development. The most updated one being agent5.
### agent5_selenium.ipynb
- Description: This Jupyter Notebook demonstrates the usage of Selenium library to create an autonomous agent for web navigation.

### agent4_selenium.ipynb
- Description: Another Jupyter Notebook showcasing the implementation of an autonomous agent using Selenium for web navigation.

### agent_1.ipynb
- Description: This Jupyter Notebook contains an implementation of the initial version of the autonomous agent for web navigation.

### api_requester.py
- Description: This file provides an API requester implementation for interacting with web services during web navigation tasks.

### cc_net5_tokenizer.py
- Description: The file includes the tokenizer implementation specific to the CC-Net5 model for tokenizing text input.

### cosine_similarity.py
- Description: This file contains the implementation of cosine similarity calculation, used for comparing vectors during web navigation tasks. It has been dropped in our final version.

### default_miniwob_vocab.txt
- Description: The file contains the default vocabulary for Miniwob tasks, used for training and inference purposes.

### dom_processing.py
- Description: This file provides functions for processing and manipulating Document Object Model (DOM) during web navigation.

### indices_dictionary2.json
- Description: JSON file containing indexes and mappings used during web navigation tasks.

### setup.py
- Description: This file includes setup configurations and dependencies required for the project.

### vocab.txt
- Description: The vocabulary file used for language modeling and training the autonomous agents.

Please refer to the individual files for more details on their implementation and usage.

If you have any questions or feedback, feel free to reach out. Happy exploring and navigating the web using AI!

