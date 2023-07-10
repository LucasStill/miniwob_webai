import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
#import gym
import gymnasium as gym
from cosine_similarity import EmbeddingFunction, VocabManagement
from cosine_similarity import find_closest_embeddings, embeddings2tokens
from cc_net5_tokenizer import CCNeT5Tokenizer
from dom_processing import iterate_dom, dict2html, prepare_t5_input
from inference_points import infer_remote_model, parse_t5_output_action
import pickle

model_endpoints = 'https://3317-142-112-54-19.ngrok-free.app'
t5_url = model_endpoints
cc_net5_url = model_endpoints

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def encode_tensor_to_json(tensor):
    tensor_data = tensor.tolist()
    json_data = json.dumps(tensor_data)
    return json_data

def encode_input_tensor_to_json(rgb_data, language_data, previous_action, t5_output):
    print(f'rgb_data: {rgb_data.shape}, language_data: {language_data.shape}, previous_action: {previous_action.shape}, t5_output: {t5_output.shape}')

    # put input into single dict
    payload_dict = {
        'rgb_data': rgb_data.tolist(),
        'language_data': language_data.tolist(),
        'previous_action': previous_action.tolist(),
        't5_output': t5_output.tolist()
    }
    json_data = json.dumps(payload_dict)
    return json_data
def decode_tensor_from_json(tensor_data):
    #tensor_data = json.loads(json_data)
    tensor = torch.tensor(tensor_data)
    return tensor


# turn action output into tensor length 10
def action2tensor(tokenizer, action_type, ref, keydown):
    data = []

    if action_type == 'click':
        data.append(0)
    else:
        data.append(1)

    # append ref
    if int(ref) < 0:
        data.append(0)
    elif int(ref)>500:
        data.append(500)
    else:
        data.append(int(ref))
    print(f'a keydown: {keydown}')
    tokenized_keydown = tokenizer.truncate_pad_entry(tokenizer.tokenize_string(keydown), 8)
    data += tokenized_keydown
    return np.array(data)


# Prepare our Embedding Function
embedding_fn_path = '/Users/lucas-andreithil/PycharmProjects/miniwob-plusplus/embedding_weights.pth'
vocab_path="/Users/lucas-andreithil/PycharmProjects/V-MPO_Lunarlander/vocab.txt"
tokenizer = CCNeT5Tokenizer(vocab_path=vocab_path)#, embedding_fn_path=embedding_fn_path)

def call_T5(action_history, utterance, dom):
    html_str = dict2html(dom)
    t5_input = prepare_t5_input(action_history, utterance, html_str)
    print(f't5_input: {t5_input[:150]}')

    # New endpoint
    t5_output = requests.post(model_endpoints + '/infer_t5', json=json.dumps(t5_input)).json()

    t5_action, t5_ref, t5_keydown = parse_t5_output_action(t5_output)

    return t5_action, t5_ref, t5_keydown

def convert_to_nested_dict(elements):
    # Create a dictionary to store elements by their 'ref' value
    element_dict = {element['ref']: element for element in elements}

    # Iterate through the elements to build the nested structure
    for element in elements:
        parent_ref = element['parent']
        if parent_ref != 0:
            parent_element = element_dict.get(parent_ref)
            if 'children' not in parent_element:
                parent_element['children'] = []
            if 'children' not in element:
                # Empty children field
                element['children'] = []
            parent_element['children'].append(element)

    # Find the root element (element with 'parent' = 0)
    root_element = None
    for element in elements:
        if element['parent'] == 0:
            root_element = element
            break

    return root_element

# format image (3, 210, 160) for request by normalizing it
def format_image(image):

    np_image = np.array(image)
    if np_image.shape[0] > 1 and np_image.shape[0] < 200:
        rgb_data = []
        for entry in np_image:
            if entry.shape[0] == 1:
                rgb_data.append(np.transpose(entry.squeeze(0) / 255.0, (2, 0, 1)))
            else:
                rgb_data.append(np.transpose(entry / 255.0, (2, 0, 1)))
        rgb_data = np.array(rgb_data)
    else:
        print(np.array(image).shape)
        rgb_data = np.transpose(np.array(image) / 255.0, (2, 0, 1))
    return rgb_data

def format_language_rgb_input(rgb_data, dom_data, utterance_data, task_data):
    # Put language task together
    language_input = np.concatenate((dom_data, utterance_data, task_data))
    #language_input = dom_data
    print(language_input.shape)

    # Iterate through language input and try to find if there are forbidden characters:
    for entry in language_input:
        if entry >= 1591:
            print(f'ENTRY IS NOT PRESENT: {entry} ################################################################################')

    rgb_data = torch.from_numpy(rgb_data).type(torch.float32)
    language_input = torch.from_numpy(language_input).type(torch.long)
    print(f'rgb_data: {rgb_data.shape}, language_input: {language_input.shape}')
    return rgb_data, language_input

import json

import torch.nn.functional as F
def parse_ccnet5_ouptut(tokenizer, output_tensor):
    # usqueeze
    output_tensor = output_tensor.unsqueeze(0)
    print(f'output_tensor: {output_tensor.shape}')
    action_type = output_tensor[:, 0]
    action_type = 'click' if int(action_type) == 0 else 'keydown'
    ref_embedding = output_tensor[:, 1:501]
    probabilities = F.softmax(ref_embedding, dim=1)
    predicted_ref = int(torch.argmax(probabilities, dim=1).to('cpu').item())

    keydowns = output_tensor[:, 501:].reshape(8, len(tokenizer.itos))
    found_keys = []
    for key in keydowns:
        probabilities = F.softmax(key.unsqueeze(0), dim=1)
        predicted_key = torch.argmax(probabilities, dim=1).to('cpu').item()
        found_keys.append(tokenizer.itos[predicted_key])
    keydowns = ''.join(found_keys).replace(tokenizer.padding_char, '')

    print(f'parsed ccnet5_output: {action_type}, {predicted_ref}, {keydowns}')
    return action_type, predicted_ref, keydowns

import requests

def call_ccnet(policy_type, rgb_input, doms, utterance, t5_output_tensor, previous_action, task_name):
    rgb_input = format_image(rgb_input)

    # format language input
    #print(doms)
    print(type(doms))
    if type(doms) == type({}):
        tokenized_dom = tokenizer.tokenize_dom(doms)
        tokenized_dom = np.array(tokenizer.truncate_pad_entry(tokenized_dom, 492)) # 492 is the value set for CC-NeT5 processing
        tokenized_utterance = np.array(tokenizer.truncate_pad_entry(tokenizer.tokenize_string(utterance), 16))
        tokenized_task_name = np.array(tokenizer.truncate_pad_entry(tokenizer.tokenize_string(task_name), 4))

        print(
            f'rgb_input: {rgb_input.shape}, tokenized_dom: {tokenized_dom.shape}, tokenized_utterance: {tokenized_utterance.shape}, tokenized_task_name: {tokenized_task_name.shape}')
        rgb_data, language_input = format_language_rgb_input(rgb_input, tokenized_dom, tokenized_utterance,
                                                             tokenized_task_name)
        # Unsqueeze with batch size to add first dimension
        rgb_data = rgb_data.unsqueeze(0)
        language_input = language_input.unsqueeze(0)

        # format previous action (flattened tensor size 577)
        print(f'previous_action: {previous_action}')
        previous_action = previous_action.unsqueeze(0)
        t5_output_tensor = torch.from_numpy(t5_output_tensor).unsqueeze(0)
        print(rgb_data.shape, language_input.shape, previous_action.shape, t5_output_tensor.shape)
        print(rgb_data.dtype, language_input.dtype, previous_action.dtype, t5_output_tensor.dtype)
        # ccnet5_payload = prepare_ccnet5_payload(rgb_data, language_input, previous_action, t5_output_tensor)
        encoded_payload = encode_input_tensor_to_json(rgb_data, language_input, previous_action, t5_output_tensor)
        print(f"about to make request to {model_endpoints + '/' + policy_type}")
        response = requests.post(model_endpoints + '/' + policy_type, json=encoded_payload)
        # Get final layer directly
        print(response)
        ccnet5_tensor = decode_tensor_from_json(response.json())
        print(f'tensor_value: {ccnet5_tensor}')
        return ccnet5_tensor, rgb_data, language_input, previous_action, t5_output_tensor
    else:
        # somehow it is sometimes already tokenized from the memory, so no need to process, except reshaping utterance and task anme
        tokenized_dom = doms
        tokenized_utterance = torch.from_numpy(np.array(tokenizer.truncate_pad_entry(tokenizer.tokenize_string(utterance), 16))).unsqueeze(0).repeat(tokenized_dom.shape[0], 1) if type(utterance) == type('') else utterance
        tokenized_task_name = torch.from_numpy(np.array(tokenizer.truncate_pad_entry(tokenizer.tokenize_string(task_name), 4))).unsqueeze(0).repeat(tokenized_dom.shape[0], 1) if type(task_name) == type('') else task_name
        #tokenized_utterance = torch.from_numpy(tokenized_utterance).unsqueeze(0).repeat(tokenized_dom.shape[0], 1)
        #tokenized_task_name = torch.from_numpy(tokenized_task_name).unsqueeze(0).repeat(tokenized_dom.shape[0], 1)

        print(f'rrgb_input: {rgb_input.shape}, tokenized_dom: {tokenized_dom.shape}, tokenized_utterance: {tokenized_utterance.shape}, tokenized_task_name: {tokenized_task_name.shape}')

        # Need to iterate to format them all
        rgb_data = []
        language_input = []
        for i in range(rgb_input.shape[0]):
            print(rgb_input[i].shape, tokenized_dom[i].squeeze(0).shape, tokenized_utterance[i].shape, tokenized_task_name[i].shape)
            #rgb_data_p, language_input_p = format_language_rgb_input(rgb_input[i], tokenized_dom[i].squeeze(0), tokenized_utterance[i], tokenized_task_name[i])
            rgb_data_p = torch.from_numpy(rgb_input[i]).type(torch.float32)
            language_input_p = tokenized_dom[i]#).type(torch.long)
            print(f'rgb_data_p: {rgb_data_p.shape} - {rgb_data_p.dtype}')
            print(f'language_input_p: {language_input_p.shape} - {language_input_p.dtype}')
            rgb_data.append(rgb_data_p.numpy())
            language_input.append(language_input_p.numpy())

        # Should have timestep as first dimension
        rgb_data = torch.from_numpy(np.array(rgb_data))
        language_input = torch.from_numpy(np.array(language_input)).squeeze(1)

        # format previous action (flattened tensor size 577)
        if type(previous_action) == type(''):
            if 'click' in previous_action:
                previous_action = 0
            elif 'key' in previous_action:
                previous_action = 1
            else:
                print(f'BIG PROBLEM PREVIOUS ACTION TYPE: {previous_action}')
        print(f'previous_action: {previous_action.shape}')
        print(f't5_output_tensor: {t5_output_tensor.shape}')
        previous_action = previous_action.squeeze(1)
        t5_output_tensor = t5_output_tensor.squeeze(2).squeeze(1)
        rgb_data = rgb_data.permute(0, 2, 3, 1) # seems like we need to permute here
        print(rgb_data.shape, language_input.shape, previous_action.shape, t5_output_tensor.shape)
        print(rgb_data.dtype, language_input.dtype, previous_action.dtype, t5_output_tensor.dtype)
        #ccnet5_payload = prepare_ccnet5_payload(rgb_data, language_input, previous_action, t5_output_tensor)
        encoded_payload = encode_input_tensor_to_json(rgb_data, language_input, previous_action, t5_output_tensor)
        print(f"about to make request to {model_endpoints + '/' + policy_type}")
        #print(encoded_payload)
        response = requests.post(model_endpoints + '/' + policy_type, json=encoded_payload)


        # Get final layer directly
        print(response)
        ccnet5_tensor = decode_tensor_from_json(response.json())

        # Convert the tensor data back into a PyTorch tensor
        #ccnet5_tensor = torch.tensor(tensor_data)

        print(f'tensor_value: {ccnet5_tensor.shape}')

        #ccnet5_action, ccnet5_ref, ccnet5_keydown = parse_ccnet5_ouptut(tokenizer, ccnet5_tensor)
        #print(f'ccnet5 output: {ccnet5_action}, {ccnet5_ref}, {ccnet5_keydown}')

        #return ccnet5_tensor, ccnet5_action, ccnet5_ref, ccnet5_keydown

        return ccnet5_tensor, rgb_data, language_input, previous_action, t5_output_tensor


# Update the mapping dictionary of ref with page changes
def update_ref2_random(old_dict, old_rand2ref, new_dict):
    _remvoved_refs_keys = old_dict.keys() - new_dict.keys()
    added_refs_keys = new_dict.keys() - old_dict.keys() # no use of old refs, better to keep them in old_dict

    # create subset_dictionary
    added_refs = {}
    for added_ref in added_refs_keys:
        if added_ref in old_dict.keys():
            added_refs[added_ref] = old_dict[added_ref]
        else:
            added_refs[added_ref] = new_dict[added_ref]

    # attribute random to new refs:
    added_refs_dict, added_rand2refs = randomize_ref_dict(list(added_refs.keys()))

    # Update the old dictionary with the new mapping
    old_dict.update(added_refs_dict)
    old_rand2ref.update(added_rand2refs)

    # These are the new mappings
    return old_dict, old_rand2ref

import random

# find all refs of the dom dict
def find_all_refs(dom):
    refs = []
    if 'ref' in dom.keys():
        refs.append(dom['ref'])
    if 'children' in dom.keys():
        for child in dom['children']:
            f_ref = find_all_refs(child)
            refs += f_ref
    return list(set(refs))

# provide ref list and return dictionary with randomized values
# also return dict to reconstruct it all
def randomize_ref_dict(refs):
    entries = refs.copy()
    random.shuffle(entries)
    ref_dict = {}
    for i, k in enumerate(refs):
        ref_dict[k] = entries[i]

    ref2rand = ref_dict
    rand2ref = {}
    for k in ref2rand.keys():
        rand2ref[ref2rand[k]] = k
    return ref_dict, rand2ref

# Assigns the randomized value to each ref of the dom dict
def attribute_random_refs_dom(dom, refs):
    if 'ref' in dom.keys():
        dom['ref'] = refs[dom['ref']]
    if 'children' in dom.keys():
        new_children = []
        for child in dom['children']:
            new_child = attribute_random_refs_dom(child, refs)
            new_children.append(new_child)
        dom['children'] = new_children
    return dom



def get_KL(prob1, logprob1, logprob2):
    kl = prob1 * (logprob1 - logprob2)
    return kl.sum(1, keepdim=True)


class Memory:
    def __init__(self):
        self.actions = []
        self.refs = []
        self.keydowns = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

        self.rgb_data = []
        self.language_input = []
        self.previous_actions = []
        self.t5_output_tensors = []

    def clear_memory(self):
        del self.actions[:]
        del self.refs[:]
        del self.keydowns[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

        del self.rgb_data[:]
        del self.language_input[:]
        del self.previous_actions[:]
        del self.t5_output_tensors[:]



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # Vocab Dictionary Management
        self.vocabulary = VocabManagement()

        # If need to infer policy, make API call to model running on GPU

        self.value_layer = nn.Sequential(
            nn.Linear(1+500+1591*8, 512), # old_state_values.size(1)
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self):
        raise NotImplementedError

    # Action function
    # Call the model to perform an action with action_layer by getting posterior distribution
    # Turn into Categorical (?)
    # Sample action from it
    # TODO: First sample action_type, then sample ref, then sample keydown
    #def act(self, state, memory):
    # memory, screenshot, doms, previous_action, t5_output_tensor, utterance, env_name)
    def act(self, policy_type, memory, screenshot, new_dom, previous_action, t5_output_tensor, utterance, task_name):

        # Make model prediction, returns torch([577])
        # PREDICTION PART
        #output = self.action_layer(state)
        #ccnet5_output_tensor, rgb_data, language_input, previous_action, t5_output_tensor = call_ccnet5(screenshot, new_dom, previous_action, t5_output_tensor, utterance, task_name)
        # rgb_input, doms, utterance, t5_output_tensor, previous_action, task_name
        ccnet5_output_tensor, rgb_data, language_input, previous_action, t5_output_tensor = call_ccnet(policy_type, rgb_input=screenshot, doms=new_dom, utterance=utterance, t5_output_tensor=t5_output_tensor, previous_action=previous_action, task_name=task_name)

        # a. Get action type
        action_type_probs = F.softmax(ccnet5_output_tensor[0, 0:1].detach())
        # Transform into format for categorical sampling
        action_type_probs = torch.from_numpy(np.array([action_type_probs[0], 1-action_type_probs[0]]))
        dist = Categorical(action_type_probs)
        action = dist.sample()

        # b. Get reference number
        ref_number_embeddings = ccnet5_output_tensor[0, 1:501].detach()#.numpy()
        # Perform Softmax to rank the closeness of the different embeddings together by their likelyhood in respect to the vector
        probabilities = F.softmax(ref_number_embeddings, dim=0)
        print(f'probabilities: {probabilities.shape}')
        dist = Categorical(probabilities)
        ref_number = dist.sample()

        # c. Get keydown if action type is keydown
        if action == 1:
            keydown_embeddings = ccnet5_output_tensor[0, 501:].detach()#.numpy()
            keydown_layers = keydown_embeddings.reshape(8, 1591)
            keydown = []
            for entry in keydown_layers:
                probabilities = F.softmax(entry, dim=0)
                dist = Categorical(probabilities)
                specific_token_word = dist.sample()
                keydown.append(specific_token_word)

            keydown = torch.from_numpy(np.array(keydown))
        else:
            # TODO: Add tensor equal to empty embedding
            # by default have zeros, but we should use the proper embedding corresponding to the empty string
            #keydown = torch.zeros(512)
            keydown = torch.full((8,), self.vocabulary.stoi[self.vocabulary.padding_char])


        #memory.states.append(state)
        memory.actions.append(action)
        memory.refs.append(ref_number)
        memory.keydowns.append(keydown)
        memory.logprobs.append(dist.log_prob(action)) # Todo: what to do with logprobs?

        # Append different model input tensors into memory
        memory.rgb_data.append(rgb_data)
        memory.language_input.append(language_input)
        memory.previous_actions.append(previous_action)
        memory.t5_output_tensors.append(t5_output_tensor)

        print(f'ref_number: {ref_number}')
        #ref_number = self.vocabulary.itos[ref_number.item()]
        #print(f'ref_number: {ref_number}')

        # Get exact keydown out of the tensors:
        keydown_str = []
        for item in keydown:
            keydown_str.append(self.vocabulary.itos[item.item()])
        keydown_str = ''.join(keydown_str).replace(self.vocabulary.padding_char, '')
        print(f'keydown_str: {keydown_str}')

        return action.item(), ref_number, keydown_str



    # TODO: Shapes don't have the right dimensions yet.
    # - action: [timesteps, 1]
    # - ref: [timesteps, 8]
    # - keydown: [timesteps, 512]
    def evaluate(self, policy_type, action, ref, keydown,
                 screenshot, new_dom, previous_action, t5_output_tensor, utterance, task_name):
        #output = self.action_layer(state) # only take first batch as we work in 1

        # XXX
        ccnet5_output_tensor, _rgb_data, _language_input, _previous_action, _t5_output_tensor = call_ccnet(policy_type, rgb_input=screenshot, doms=new_dom, utterance=utterance, t5_output_tensor=t5_output_tensor, previous_action=previous_action, task_name=task_name)

        #call_ccnet5(screenshot, new_dom, previous_action, t5_output_tensor, utterance, task_name)


        # a. Action type
        # Create distribution for all action types by iterating as we need to create the binary distribution
        action_probs = []
        for ac in ccnet5_output_tensor[:, 0:1]:#.detach().numpy():
            print(f'AC: {ac[0]}')
            #ac[0] = F.softmax(ac[0], dim=0)
            print(f'AC2: {ac[0]}')
            pp = np.array([ac[0], 1 - ac[0]])

            min_value = np.min(pp)
            max_value = np.max(pp)

            # Perform min-max normalization
            action_binary_prob = (pp - min_value) / (max_value - min_value)

            if action_binary_prob[0] == 1:
                action_binary_prob[0] = 0.90
                action_binary_prob[1] = 0.10
            elif action_binary_prob[1] == 1:
                action_binary_prob[1] = 0.90
                action_binary_prob[0] = 0.10

            print(f'action_binary_prob: {action_binary_prob}')
            action_probs.append(action_binary_prob)
        action_probs = torch.from_numpy(np.stack(action_probs))
        print(f'action_probs: {action_probs}')
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        print(f'action_logprobs: {action_logprobs}')
        action_dist_probs = dist.probs
        print(f'action_dist_probs: {action_dist_probs}')

        # b. Reference Number
        ref_number_embeddings = ccnet5_output_tensor[:, 1:501]#.detach().numpy()
        ref_probs = []
        for ref_o in ref_number_embeddings:#.detach().numpy():
            #print(f'ref_o: {ref_o.shape}')
            probabilities = F.softmax(ref_o, dim=0)
            ref_probs.append(probabilities)
        ref_probs = torch.from_numpy(np.stack(ref_probs))
        dist = Categorical(ref_probs)
        # todo: check type and shape of ref into dist.log_probs
        ref_logprobs = dist.log_prob(ref)
        ref_dist_probs = dist.probs


        # c. Keydown Text
        keydown_output = ccnet5_output_tensor[:, 501:]
        keydown_probs = []
        # iterate through batch
        for keydown_o in keydown_output:#.detach().numpy():
            keydown_layers = keydown_o.reshape(8, 1591)
            level_distributions = []
            for entry in keydown_layers:
                probabilities = F.softmax(entry, dim=0)
                level_distributions.append(list(probabilities))
            keydown_probs.append(level_distributions)
        keydown_probs = torch.from_numpy(np.array(keydown_probs))

        # Iterate through the chunks of the second dimension to create the individual Distribution over past experience

        keydown_logprobs = []
        keydown_dist_probs = []
        for i, chunk in enumerate(keydown_probs.chunk(keydown_probs.size(1), dim=1)):
            # We must iterate through the different created distribution and create 8 different Categorical for each keydown part
            dist = Categorical(chunk)
            # Make dist_log_prob over the keydown at specific embeddings
            positional_embedding = keydown[:, i:i+1]
            keydown_logprobs_pos = dist.log_prob(positional_embedding)
            keydown_dist_probs_pos = dist.probs
            keydown_logprobs.append(keydown_logprobs_pos)
            keydown_dist_probs.append(keydown_dist_probs_pos)


        # Make inference on value_layer
        # TODO: as single or divide action, ref and keydown?
        #state_value = self.value_layer(state)
        # we don't save the transformed input here
        ccnet5_output_tensor, _rgb_data, _language_input, _previous_action, t5_output_tensor = call_ccnet(policy_type=policy_type, rgb_input=screenshot, doms=new_dom, utterance=utterance, t5_output_tensor=t5_output_tensor, previous_action=previous_action, task_name=task_name)

        # Pass to value layer
        state_value = self.value_layer(ccnet5_output_tensor)

                              #todo: For this one here below
        return action_logprobs, torch.squeeze(state_value), action_dist_probs, ref_logprobs, ref_dist_probs, keydown_logprobs, keydown_dist_probs


class VMPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.eta = torch.autograd.Variable(torch.tensor(1.0), requires_grad=True)
        self.alpha = torch.autograd.Variable(torch.tensor(0.1), requires_grad=True)
        self.eps_eta = 0.02
        self.eps_alpha = 0.1

        # Todo: get parameters of Policy
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)

        #params = [
        #    #{'params': self.policy.parameters()},
        #    {'params': requests.post(model_endpoints + '/get_policy_parameters', json=json.dumps('')).json()},
        #    {'params': self.eta},
        #    {'params': self.alpha}
        #]

        #self.optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=1e-1)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory, utterance, task):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_refs = torch.stack(memory.refs).to(device).detach()
        old_keydowns = torch.stack(memory.keydowns).to(device).detach()
        # old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Old States
        old_rgb = torch.stack(memory.rgb_data).to(device).detach()
        old_language_input = torch.stack(memory.language_input).to(device).detach()
        old_previous_actions = torch.stack(memory.previous_actions).to(device).detach()
        old_t5_out_tensors = torch.stack(memory.t5_output_tensors).to(device).detach()

        print(f'old_language_input: {old_language_input.shape}')
        # get old probs and old advantages
        with torch.no_grad():
            # TODO: separate them, or keep them all as one?
            _, old_state_values, old_action_dist_probs, _, old_ref_dist_probs, _, old_keydown_dist_probs = self.policy_old.evaluate('make_inference_old', old_actions, old_refs, old_keydowns,
                                                                                                                               old_rgb, old_language_input, old_previous_actions, old_t5_out_tensors, utterance, task)
            # test passing a linear layer
            # todo: improve value function as in the paper

            #value_state_passed = value_layer(old_state_values).squeeze(1)
            print(f'value_state_passed: {old_state_values.shape}')
            print(f'rewards: {rewards.shape}')
            advantages = rewards - old_state_values #old_state_values.detach()
            print(f'advantages: {advantages.shape}')

        # Optimize policy for K epochs:
        for i in range(self.K_epochs):
            # Evaluating sampled actions and values :
            # TODO: separate them, or keep them all as one?
            action_logprobs, state_values, action_dist_probs, ref_logprobs, ref_dist_probs, keydown_logprobs, keydown_dist_probs = self.policy.evaluate('make_inference_normal', old_actions, old_refs, old_keydowns,
                                                                                                                                                        old_rgb, old_language_input, old_previous_actions, old_t5_out_tensors, utterance, task)
            # todo: need to check if this is the right use of the value layer
            #state_values = value_layer(state_values).squeeze(1) #(steps, 1) to (steps)
            print(f'state_values though value layer: {state_values.shape}')

            # Run V-MPO loss computation algorithm
            # TODO: Get loss for one of the elements and then apply our adaptation to it by weighting it?
            def compute_vmpo_loss(old_dist_probs, logprobs, distprobs):
                # Get samples with top half advantages
                advprobs = torch.stack((advantages, logprobs))
                advprobs = advprobs[:, torch.sort(advprobs[0], descending=True).indices]
                good_advantages = advprobs[0, :len(old_language_input) // 2] # We just care about the length, not the shape
                good_logprobs = advprobs[1, :len(old_language_input) // 2]

                # Get losses
                phis = torch.exp(good_advantages / self.eta.detach()) / torch.sum(
                    torch.exp(good_advantages / self.eta.detach()))
                L_pi = -phis * good_logprobs
                L_eta = self.eta * self.eps_eta + self.eta * torch.log(torch.mean(torch.exp(good_advantages / self.eta)))

                KL = get_KL(old_dist_probs.detach(), torch.log(old_dist_probs).detach(), torch.log(distprobs))

                L_alpha = torch.mean(self.alpha * (self.eps_alpha - KL.detach()) + self.alpha.detach() * KL)

                loss = L_pi + L_eta + L_alpha + 0.5 * self.MseLoss(state_values, rewards)
                return loss, KL

            # Compute the losses for each target type through V-MPO
            print(f'advantages: {advantages.shape}')
            print(f'old_action_dist_probs: {old_action_dist_probs.shape}, action_logprobs: {action_logprobs.shape}, action_dist_probs: {action_dist_probs.shape}')
            print(f'old_action_dist_probs: {old_action_dist_probs}, action_logprobs: {action_logprobs}, action_dist_probs: {action_dist_probs}')
            loss_action, KL_action = compute_vmpo_loss(old_action_dist_probs, action_logprobs, action_dist_probs)
            loss_ref, KL_ref = compute_vmpo_loss(old_ref_dist_probs, ref_logprobs, ref_dist_probs)

            # Need to iterate through the Keydown Dists as they are a nested list of Distributions
            avg_keydown_loss = torch.zeros_like(loss_ref)
            avg_KL_keydown = torch.zeros_like(KL_ref)
            for old_k_dist_probs, k_logprobs, k_dist_probs in zip(old_keydown_dist_probs, keydown_logprobs, keydown_dist_probs):
                loss_keydown, KL_keydown = compute_vmpo_loss(old_k_dist_probs.squeeze(), k_logprobs.squeeze(), k_dist_probs.squeeze())
                avg_keydown_loss += loss_keydown
                avg_KL_keydown += KL_keydown
            avg_keydown_loss = avg_keydown_loss / 8
            avg_KL_keydown = avg_KL_keydown / 8
            # Weight the losses together
            print(f'loss_action: {loss_action}, loss_ref: {loss_ref}, avg_keydown_loss: {avg_keydown_loss}')
            loss = 1/3 * loss_action + 1/3 * loss_ref + 1/3 * avg_keydown_loss
            print(f'LOSS = {loss.shape}, {loss}     ######################################')

            # take gradient step
            #self.optimizer.zero_grad()
            #loss.mean().backward()
            #self.optimizer.step()

            # backprop to remote gpu
            encoded_payload = encode_tensor_to_json(loss.mean())
            response = requests.post(model_endpoints + '/backprop_loss_model', json=encoded_payload)
            print(f'response backprop: {response.json()}')

            with torch.no_grad():
                self.eta.copy_(torch.clamp(self.eta, min=1e-8))
                self.alpha.copy_(torch.clamp(self.alpha, min=1e-8))
            if i == self.K_epochs - 1:
                print(f'KL_action={torch.mean(KL_action).item()}, KL_ref={torch.mean(KL_ref).item()}, KL_keydown={torch.mean(avg_KL_keydown).item()}, self.alpha.item()')
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

from miniwob.action import ActionTypes
import pandas as pd

def main():
    task_names = [
        'book-flight',
        'chose-date-easy',
        'choose-date-medium',
        'choose-date',
        'choose-list',
        'click-button-sequence',
        'click-button',
        'click-checkboxes-soft',
        'click-checkboxes-transfer',
        'click-checkboxes',
        'click-collapsible-2',
        'click-collapsible',
        'click-color',
        'click-dialog-2',
        'click-dialog',
        'click-link',
        'click-menu-2',
        'click-menu',
        'click-option',
        'click-pie',
        'click-scroll-list',
        'click-shades',
        'click-shape',
        'click-tab-2-easy',
        'click-tab',
        'click-test-2',
        'click-test-transfer',
        'click-widget',
        'count-shape',
        'enter-text',
        'count-shape',
        'email-inbox-forward-nl-turk',
        'email-inbox-forward-nl',
        'email-inbox',
        'enter-date',
        'enter-password',
        'enter-text-2',
        'enter-text-dynamic',
        'enter-text',
        'enter-time',
        'focus-text-2',
        'focus-text',
        'grid-coordinate',
        'guess-number',
        'identify-shape',
        'login-user-popup',
        'login-user',
        'multi-layouts',
        'multi-orderings',
        'navigate-tree',
        'Search-engine',
        'social-media-all',
        'social-media-some',
        'social-media',
        'tic-tac-toe',
        'use-autocomplete',
        'use-spinner',
    ]


    ############## Hyperparameters ##############
    #state_dim = env.observation_space.shape[0]
    state_dim = 512
    action_dim = 1+500+1591*8 # Size of target output
    ref_dim = 500 # 64 Size embedding for the reference number
    keydown_dim = 8*1591 # 8x64 size embedding for the target keydown word. Or should we predict an index instead?
    render = False
    solved_reward = 230  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 4  # max training episodes
    max_timesteps = 8  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 5 #2400  # update policy every n timesteps
    lr = 1e-4
    betas = (0.9, 0.999)
    gamma = 0.9  # discount factor
    K_epochs = 8  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    #############################################


    memory = Memory()
    ppo = VMPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0


    # Dataset recording successful episodes
    successful_episodes = []

    error_logs = []

    env_name = "miniwob/click-test-2-v1"
    # creating environment

    last_successful_episodes_len = 0
    #successful_episodes_df = [] # just default to get erased
    df = pd.read_csv('/Users/lucas-andreithil/PycharmProjects/miniwob-plusplus/successful_episodes_df.csv')
    successful_episodes_df = df.values.tolist()

    max_episodes_tasks = 10000000
    for i_task in range(max_episodes_tasks):
        print(f'@@@@@@@@@ STARTING EPISODE {i_task}')

        # Loop to revocver
        try:
            #if True:
            print(f'####### Starting episode random {i_task}')
            env_name = random.choice(task_names)
            try:
                env = gym.make('miniwob/' + env_name)
            except Exception as e:
                print(f'env name not found: {env_name}')

            if random_seed:
                torch.manual_seed(random_seed)
                env.seed(random_seed)


            # training loop
            # Should have a max amount of tries per task before moving to a next one
            for i_episode in range(1, max_episodes + 1):

                # Saves episode if found new succesful one
                if len(successful_episodes) > 0 and len(successful_episodes_df) >= last_successful_episodes_len:
                    print(f'found successful_episodes: {len(successful_episodes)}')
                    successful_episodes_df = pd.DataFrame(successful_episodes)
                    print(successful_episodes_df.head())
                    successful_episodes_df.to_csv('successful_episodes_df.csv', index=False)
                    last_successful_episodes_len = len(successful_episodes_df)
                    print(f'found successful_episodes: {len(successful_episodes)}')
                    successful_episodes_df = pd.DataFrame(successful_episodes)
                    print(successful_episodes_df.head())
                    successful_episodes_df.to_csv('successful_episodes_df.csv', index=False)

                # Episode Data
                dom_episode = []
                action_history = []
                # Set up previous_action with default values
                previous_action = torch.zeros(10)

                # Empty ref dictionaries at start of episode:
                ref_dict, rand2ref = {}, {}

                # Records episode data
                successful_episodes_dom = []
                successful_episodes_screenshot = []
                successful_episodes_action = []
                successful_episodes_ref = []
                successful_episodes_keydown = []
                successful_t5_output = []

                # Initial observation of environment
                observation, info = env.reset()

                for t in range(max_timesteps):
                    timestep += 1

                    # Make observation here: observation at previous timestep is done lower in this loop
                    # 1: Reformat Screenshot Shape
                    screenshot = observation['screenshot']
                    print(f'screenshot shape: {screenshot.shape}')

                    # 2: Check Utterance
                    utterance = observation['utterance']

                    # 3: Process DOM
                    # convert elements to nested dict structure
                    nested_dict = convert_to_nested_dict(observation['dom_elements'])
                    # clean
                    _, _, new_dom = iterate_dom(nested_dict)
                    # randomize references
                    ref_dict_new, _rand2ref_new = randomize_ref_dict(find_all_refs(new_dom))
                    # update previous dictionaries with new observations
                    ref_dict, rand2ref = update_ref2_random(ref_dict, rand2ref, ref_dict_new)
                    # apply randomized references to observation DOM
                    doms = attribute_random_refs_dom(new_dom, ref_dict)

                    # Running policy_old in calling Frozen T5:
                    t5_action, t5_ref, t5_keydown = call_T5(action_history, utterance, doms)
                    t5_output_tensor = action2tensor(tokenizer, t5_action, t5_ref, t5_keydown)

                    # Then calling the CC-Net based architecture in the PPO policy including the T5 outputs
                    print(f'previous_action main loop: {previous_action.shape}')
                    action, ref, keydown = ppo.policy_old.act('make_inference_old', memory, screenshot, doms, previous_action, t5_output_tensor, utterance, env_name)
                    print(f'PPO POLICY ACTION: {action}, ref: {ref}, keydown: {keydown}')
                    # TODO: get correct action space

                    if action == 0:
                        action_type = ActionTypes.CLICK_ELEMENT
                    else:
                        action_type = ActionTypes.TYPE_TEXT

                        # Sample and Create Action
                    action = env.action_space.sample()  # Template for the action.
                    action["action_type"] = env.action_space_config.action_types.index(
                        action_type
                    )
                    # Get the ref number by using the dictionary mapping random to old reference
                    # TODO: do the randomized mapping of refs here
                    try:
                        print(f'attributing ref {ref}')
                        action["ref"] = str(rand2ref[int(ref)])
                    except Exception as e:
                        print(e)
                        print(rand2ref)
                        print(f'ref: {ref}')
                        print(f'REF VALUE IS NOT PRESENT IN DOM DICTIONARY: {ref}')
                        print(f'Attributing default ref value...')
                        action["ref"] = str('4')

                    # Add text if action is type_text
                    if action_type == ActionTypes.TYPE_TEXT:
                        action['text'] = keydown.replace(tokenizer.padding_char, '')

                    # Perform action, get observation for eventual next part of the loop
                    observation, reward, terminated, truncated, info = env.step(action)

                    # Recoding data
                    successful_episodes_dom.append(doms)
                    successful_episodes_screenshot.append(screenshot)
                    successful_episodes_action.append('click' if action['action_type'] == 8 else 'keydown')
                    successful_episodes_ref.append(int(action['ref']))
                    successful_episodes_keydown.append(keydown)
                    successful_t5_output.append([t5_action, t5_ref, t5_keydown])

                    # Saving reward and is_terminal:
                    memory.rewards.append(reward)
                    memory.is_terminals.append(terminated)

                    # ACTION HISTORY
                    # important to translate the action type below
                    action_type = 0 if action['action_type'] == 8 else 1
                    keydown = '' if action['text'] == 'q[]' or action_type == 0 else action['text'].replace('q[', '').replace(']', '')
                    # todo: use ref_dict to translate previous action_ref
                    action_history.append([action_type, ref_dict[int(action['ref'])], keydown])

                    # Update the previous action with this one
                    previous_action = torch.from_numpy(action2tensor(tokenizer, action, ref, keydown))

                    # update if its time
                    if timestep % update_timestep == 0:
                        ppo.update(memory, utterance, env_name)
                        memory.clear_memory()
                        timestep = 0

                    running_reward += reward
                    if render or running_reward > (log_interval * solved_reward) * 0.8:
                        env.render()
                    if terminated or reward < 0:
                        print(f'EPISODE TERMINATED')

                        if reward > 0:
                            states = {
                                'successful_episodes_dom': successful_episodes_dom,
                                'successful_episodes_screenshot': successful_episodes_screenshot,
                                'successful_episodes_action': successful_episodes_action,
                                'successful_episodes_ref': successful_episodes_ref,
                                'successful_episodes_keydown': successful_episodes_keydown,
                                'successful_t5_output': successful_t5_output
                            }
                            t_name = env_name.split('/')
                            row = {
                                'task_name': t_name[len(t_name)-1],
                                'utterance': utterance,
                                'states': states,
                                'reward': reward
                            }
                            successful_episodes.append(row)

                        break

                    # to remove
                    #break

                avg_length += t

                # stop training if avg_reward > solved_reward
                if running_reward > (log_interval * solved_reward):
                    print("########## Solved! ##########")
                    torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
                    break

                # logging
                if i_episode % log_interval == 0:
                    avg_length = int(avg_length / log_interval)
                    running_reward = int((running_reward / log_interval))

                    print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
                    running_reward = 0
                    avg_length = 0

                # to remove
                if i_episode > 1 and False:
                    break
        #if False:
        except Exception as e:
            print(f'Error Exception outer loop, trying to recover in two seconds')
            time.sleep(2)
            if len(error_logs)<700:
                error_logs.append(e)
            continue


    print(f'found successful_episodes: {len(successful_episodes)}')
    successful_episodes_df = pd.DataFrame(successful_episodes)
    print(successful_episodes_df.head())
    successful_episodes_df.to_csv('successful_episodes_df.csv', index=False)

if __name__ == '__main__':
    main()
