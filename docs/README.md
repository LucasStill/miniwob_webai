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




## Training files, prototypes and remote inference notebooks:

### LoadT5 Test.ipynb
- Description: This Jupyter Notebook is used for testing the loading and implementation of the T5 model for web navigation tasks.

### Model Inference CC_NeT5.ipynb
- Description: This Jupyter Notebook showcases the implementation of model inference for the CC_NeT5 model, enabling the autonomous agent to make predictions during web navigation.

### Selenium agent.ipynb
- Description: This Jupyter Notebook tests the use of Selenium library to create an autonomous agent for web navigation.

### T5 Last action clean.ipynb
- Description: This Jupyter Notebook focuses on cleaning and processing the last action data for the T5 model during web navigation tasks.

### T5 Training.ipynb
- Description: This Jupyter Notebook provides the implementation of training the T5 model for web navigation tasks.

### T5Train2.ipynb
- Description: This Jupyter Notebook presents an alternative implementation of training the T5 model, exploring different techniques and approaches.

### T5_Dataset.ipynb
- Description: This Jupyter Notebook includes the creation and handling of datasets specifically tailored for training the T5 model for web navigation.

### T5_Training.ipynb
- Description: Another Jupyter Notebook dedicated to the training of the T5 model, focusing on optimizing its performance for web navigation tasks.

### T5_hierarchy.ipynb
- Description: This Jupyter Notebook explores the hierarchical structure of the T5 model and its implications for web navigation tasks.

### T5_hierarchy2.ipynb
- Description: An additional Jupyter Notebook investigating the hierarchical aspects of the T5 model for improved web navigation.

### Task Indices Layout.ipynb
- Description: This Jupyter Notebook provides the layout and organization of task indices used during web navigation tasks.

### finetune_miniwob.ipynb
- Description: This Jupyter Notebook focuses on fine-tuning the T5 model specifically for Miniwob tasks during web navigation, initial test over miniwob.

### finetuning_miniwob2.ipynb
- Description: Another Jupyter Notebook dedicated to the fine-tuning of the T5 model for Miniwob tasks, exploring different techniques.

### test_metrics.ipynb
- Description: This Jupyter Notebook focuses on testing and evaluating the performance metrics of the trained models during web navigation tasks. We prototype different metrics.

### trainminiwob3.ipynb
- Description: This Jupyter Notebook provides the implementation of training the autonomous agents specifically for Miniwob tasks.

## Miniwob
Miniwob parameters to run the environments are included in the /miniwob folder and makes uses of the /viewer directory. Credits go to the Farama foundation with the original repository located at https://github.com/Farama-Foundation/miniwob-plusplus/tree/master 

## Screenshots for training
The recorded screenshots for the multimodal approach can be found in a separate repository at https://github.com/LucasStill/miniwob_zip

## Weights
### SL weights
The weights of the combined approach for T5H-large and the CC-Net inspired architecture after the first SL phase can be found here: https://drive.google.com/file/d/1XfK8e-jBrROShOmY2nBx-VQh3CC7IAS9/view?usp=sharing

### RL weights
The weights regarding the final phase RL can be found here: https://drive.google.com/file/d/1ty4pBBQvv4b7JzDNL8Z1dv1bPICJmsRM/view?usp=drive_link


Please refer to the individual files for more details on their implementation and usage.

If you have any questions or feedback, feel free to reach out. Happy exploring and navigating the web using AI!
