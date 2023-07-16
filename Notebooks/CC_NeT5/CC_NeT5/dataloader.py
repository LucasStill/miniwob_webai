

# Load some of the images we have here and try to save them to the dataset
import os
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import pandas as pd
import re
import json
from torch.utils.data import DataLoader


class DatasetLoader:
    def __init__(self, screenshots_path, dataset, start_index, end_index, dom_tokenizer, batch_size):
        # Get the Subset of the dataset
        self.start_index = start_index
        self.end_index = end_index
        self.df = dataset['train'].to_pandas()[self.start_index:self.end_index]
        # When creating the class, parse JSON files into a state-based list structure
        self.df['processed_states'] = self.df['processed_states'].apply(lambda x: json.loads(re.sub(r'\b(True|False)\b', lambda m: m.group(0).lower(), x.replace("'", '"'))))
        print(f'Dataset subset has {len(self.df)} rows')

        self.screenshots_path = screenshots_path
        self.indices, found_screenshots = self.get_file_indices()
        print(f'Found a total of {found_screenshots} state screenshots for {len(self.indices)} episodes.')

        self.add_images_to_episodes()

        self.dom_tokenizer = dom_tokenizer
        self.dom_seq_length = 492
        self.utterance_seq_length = 16
        self.task_name_seq_length = 4

        self.target_action_type_seq_length = 1
        self.target_ref_seq_length = 1
        self.target_keydown_seq_length = 8 #32

        # Size of the final layer output being composed of
        # 1 (action),
        # + 64 (ref) (
        # + 64x8 (keydown text times target_keydown_seq_length)
        self.model_output_size = 577

        self.batch_size = batch_size

    # Add the images to the different episodes of the dataset
    def add_images_to_episodes(self):
        found_images = []
        i = self.start_index
        image_counter = 0
        while i < self.end_index:
            # Get K indices
            images = []
            state_indexes = self.indices[i]
            for state_index in state_indexes:
                images.append(self.read_image(i, state_index))
                image_counter += 1
            found_images.append(images)
            i += 1
        # Add new column
        self.df['episode_images'] = found_images
        print(f'Found total of {image_counter} images in the loaded episodes.')

    def get_file_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for filename in os.listdir(self.screenshots_path):
            if filename.endswith('.png'):
                # Extract the N and K values from the filename
                N, K = filename.replace('sc_', '').replace('st_', '').replace('.png', '').split('_')
                N = int(N)
                K = int(K)
                indices.append((N, K))

        # Turn the indices into a dictionary
        indices_dict = {}
        for (N, K) in indices:
            if N not in indices_dict:
                indices_dict[N] = [K]
            else:
                indices_dict[N].append(K)
                indices_dict[N] = sorted(indices_dict[N])

        return indices_dict, len(indices)

    import torch

    # Reads images into tensor
    def read_image(self, N, K) -> torch.Tensor:
        filename = f'sc_{N}_st_{K}.png'
        filepath = os.path.join(self.screenshots_path, filename)
        image = Image.open(filepath).convert('RGB')
        tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return tensor

    def save_image_png(self, N, K):
        loaded_image = self.read_image(N, K).numpy()
        print(f'Image has shape {loaded_image.shape}')

        # scale the pixel values from [0,1] to [0, 255]
        array = np.clip(loaded_image * 255, 0, 255).astype('uint8')
        img = Image.fromarray(array.transpose(1, 2, 0), mode='RGB')
        img.save('screenshots/test_img.png')

    # Turn our dataset into a series of state tensors ready for training.
    # We though ensure to clearly separate our different instances:
    # - todo: T5-Data (for later, depends between SL or RL current approach)
    # - RBG
    # - todo: Tokenized DOM
    # - todo: Task Instruction
    # They are separated because they are not going to be fed in the same manner into the model.
    # Basically into one single row.
    # todo: skip rows that have negative rewards
    def process_dataset(self):
        rbg_data = []
        dom_data = []
        utterance_data = []
        task_name_data = []

        target_action = []
        target_refs = []
        target_keydown = []

        episode_previous_actions = []

        for (index, row) in self.df.iterrows():
            if float(row['reward']) <= 0: # Skips failed episodes
                continue
            if len(row['episode_images']) != len(row['processed_states']): # Skips episodes with no matching pictures amount and states
                print(f'Error: episode_images and processed_states don\' have the same length for index {index}')
                continue
            for rbg in row['episode_images']:
                rbg = np.array(rbg, dtype='float64')
                print(f'rbg shape: {rbg.shape}')
                rbg_data.append(rbg)


            # Process DOM Data
            for (index_state, state) in enumerate(row['processed_states']):

                # Output Tokens
                if state['action_type'] == 'click': # Append boolean for action click or keydown
                    t_action = np.array([0])
                    target_action.append(t_action)
                else:
                    t_action = np.array([1])
                    target_action.append(t_action)

                # Tokenize ref and appends to target
                t_ref = np.array(self.dom_tokenizer.truncate_pad_entry(self.dom_tokenizer.tokenize_string(str(state['refs'])), self.target_ref_seq_length))
                target_refs.append(t_ref)
                # Tokenize target text and appends to target
                t_keydown = np.array(self.dom_tokenizer.truncate_pad_entry(self.dom_tokenizer.tokenize_string(state['keydown_text']), self.target_keydown_seq_length))
                target_keydown.append(t_keydown)

                # Input Tokens
                # Tokenize dom
                tokenized_dom = self.dom_tokenizer.tokenize_dom(state['dom'])
                tokenized_dom = self.dom_tokenizer.truncate_pad_entry(tokenized_dom, self.dom_seq_length)
                dom_data.append(tokenized_dom)

                # Tokenize utterance
                tokenized_utterance = self.dom_tokenizer.truncate_pad_entry(self.dom_tokenizer.tokenize_string(row['utterance']), self.utterance_seq_length)
                utterance_data.append(tokenized_utterance)

                # Tokenize task name
                tokenized_task_name = self.dom_tokenizer.truncate_pad_entry(self.dom_tokenizer.tokenize_string(row['task_name']), self.task_name_seq_length)
                task_name_data.append(tokenized_task_name)

                # Deal with the previous action
                if index_state < len(row['processed_states'])-1:
                    # Default one so empty for the first action
                    if index_state == 0:
                        episode_previous_actions.append([])
                        #episode_previous_actions.append(self.dom_tokenizer.truncate_pad_entry(self.dom_tokenizer.tokenize_string(''), self.model_output_size))

                    # Add previous actions one by one
                    episode_previous_actions.append([t_action, t_ref, t_keydown])
                # Else no action to add




        rbg_data = np.stack(rbg_data)
        dom_data = np.array(dom_data)
        utterance_data = np.array(utterance_data)
        task_name_data = np.array(task_name_data)
        target_action = np.array(target_action)
        target_refs = np.array(target_refs)
        target_keydown = np.array(target_keydown)


        return rbg_data, dom_data, utterance_data, task_name_data, target_action, target_refs, target_keydown, episode_previous_actions


    # Batchify the dataset
    def get_dataloder(self, dataset):
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader


def test_conct():

    actions = torch.randn(10, 1)
    print(actions.shape)

    keydowns = torch.randn(10, 8, 64)
    print(keydowns.shape)

    conct = torch.cat((actions, keydowns), dim=1)