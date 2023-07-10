import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
#import gym
import gymnasium as gym
from cosine_similarity import EmbeddingFunction, VocabManagement
from cosine_similarity import find_closest_embeddings, embeddings2tokens
from cc_net5_tokenizer import CCNeT5Tokenizer
from dom_processing import iterate_dom, dict2html, prepare_t5_input
from inference_points import infer_remote_model, parse_t5_output_action
import pickle

t5_url = 'https://6ce4-34-147-56-214.ngrok-free.app'
cc_net5_url = 'https://1481-35-226-96-204.ngrok-free.app'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def call_ccnet5(screenshot, new_dom, previous_action, t5_output_tensor, utterance, task_name):
    # format RGB input
  rgb_input = format_image(screenshot)

  # format language input
  tokenized_dom = tokenizer.tokenize_dom(new_dom)
  tokenized_dom = np.array(tokenizer.truncate_pad_entry(tokenized_dom, 492)) # 492 is the value set for CC-NeT5 processing
  tokenized_utterance = np.array(tokenizer.truncate_pad_entry(tokenizer.tokenize_string(utterance), 16))
  tokenized_task_name = np.array(tokenizer.truncate_pad_entry(tokenizer.tokenize_string(task_name), 4))
  rgb_data, language_input = format_language_rgb_input(rgb_input, tokenized_dom, tokenized_utterance, tokenized_task_name)
  # Unsqueeze with batch size to add first dimension
  rgb_data = rgb_data.unsqueeze(0)
  language_input = language_input.unsqueeze(0)

  # format previous action (flattened tensor size 577)
  previous_action = previous_action.unsqueeze(0)
  t5_output_tensor = t5_output_tensor.unsqueeze(0)

  # Inference to model CC-NeT5
  ccnet5_payload = prepare_ccnet5_payload(rgb_data, language_input, previous_action, t5_output_tensor)
  # TODO: make request to return full 577 tensor
  ccnet5_output_tensor = infer_remote_model(ccnet5_payload, cc_net5_url + '/infer_actor_normal')['prediction']

  values_string = ccnet5_output_tensor.replace('tensor(', '').replace(')', '').split(']]')[0]
  values_string = values_string[2:]
  values_list = [float(value) for value in values_string.split(',')]
  ccnet5_output_tensor = torch.tensor(values_list).view(1, -1).squeeze()
  # Deserialize the loss function
  #ccnet5_output_tensor = pickle.loads(ccnet5_output_tensor.encode('latin1'))

  print(f'ccnet5_output_tensor: {ccnet5_output_tensor.shape}')


  # We also return other previous tensors to add to the memory after they have been tokenizer
  return ccnet5_output_tensor, rgb_data, language_input, previous_action, t5_output_tensor



# Prepare our Embedding Function
embedding_fn_path = '/Users/lucas-andreithil/PycharmProjects/miniwob-plusplus/embedding_weights.pth'
vocab_path="/Users/lucas-andreithil/PycharmProjects/V-MPO_Lunarlander/vocab.txt"
tokenizer = CCNeT5Tokenizer(vocab_path=vocab_path, embedding_fn_path=embedding_fn_path)

def call_T5(action_history, utterance, dom):
    html_str = dict2html(dom)
    t5_input = prepare_t5_input(action_history, utterance, html_str)
    print(f't5_input: {t5_input[:50]}')
    t5_output = infer_remote_model(t5_input, t5_url)
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
    rgb_data = np.transpose(np.array(image) / 255.0, (2, 0, 1))
    return rgb_data

def format_language_rgb_input(rgb_data, dom_data, utterance_data, task_data):
    # Put language task together
    language_input = np.concatenate((dom_data, utterance_data, task_data))
    #language_input = dom_data
    print(language_input.shape)

    rgb_data = torch.from_numpy(rgb_data).type(torch.float32)
    language_input = torch.from_numpy(language_input).type(torch.long)
    language_input = tokenizer.embedding_fn(language_input) # Create embeddings for language
    print(f'rgb_data: {rgb_data.shape}, language_input: {language_input.shape}')
    return rgb_data, language_input

import json

def prepare_ccnet5_payload(rgb_data, language_input, previous_action, t5_output_tensor):
    cc_net5_payload = {
        'rgb_input': rgb_data,
        'language_input': language_input,
        'previous_action': previous_action,
        't5_output': t5_output_tensor,
    }

    cc_net5_payload_numpy = {
        key: value.detach().numpy().tolist() if isinstance(value, torch.Tensor) else value
        for key, value in cc_net5_payload.items()
    }

    json_data = json.dumps(cc_net5_payload_numpy)
    return json_data


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
    def __init__(self, state_dim, action_dim, n_latent_var, embedding_fn):
        super(ActorCritic, self).__init__()

        # Embedding function for the representation
        self.embedding_fn = embedding_fn

        # Vocab Dictionary Management
        self.vocabulary = VocabManagement()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        # find out what the critic is doing
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    # Action function
    # Call the model to perform an action with action_layer by getting posterior distribution
    # Turn into Categorical (?)
    # Sample action from it
    # TODO: First sample action_type, then sample ref, then sample keydown
    #def act(self, state, memory):
    def act(self, memory, screenshot, new_dom, previous_action, t5_output_tensor, utterance, task_name):

        # Make model prediction, returns torch([577])
        # PREDICTION PART
        #output = self.action_layer(state)
        ccnet5_output_tensor, rgb_data, language_input, previous_action, t5_output_tensor = call_ccnet5(screenshot, new_dom, previous_action, t5_output_tensor, utterance, task_name)

        # a. Get action type
        action_type_probs = ccnet5_output_tensor[0:1].detach().numpy()
        # Transform into format for categorical sampling
        action_type_probs = torch.from_numpy(np.array([1 - action_type_probs[0], action_type_probs[0]]))
        dist = Categorical(action_type_probs)
        action = dist.sample()

        # b. Get reference number
        ref_number_embeddings = ccnet5_output_tensor[1:65].detach().numpy()
        # Perform cosine similarity to rank the closeness of the different embeddings together by their likelyhood in respect to the vector
        _tokens, ref_cosine_likelyhood = embeddings2tokens(ref_number_embeddings, self.embedding_fn)
        ref_cosine_likelyhood = torch.softmax(torch.from_numpy(ref_cosine_likelyhood[0]), dim=0).squeeze()
        dist = Categorical(ref_cosine_likelyhood)
        ref_number = dist.sample()

        # c. Get keydown if action type is keydown
        if action == 1:
            keydown_embeddings = ccnet5_output_tensor[65:577].detach().numpy()
            # TODO: Perform Cosine Similarity and sample from the obtained indexes
            #keydown_cosine_likelyhood = find_closest_embeddings(keydown_probs, self.embedding_fn.embedding.weight.detach().numpy())
            _tokens, keydown_cosine_likelyhood = embeddings2tokens(keydown_embeddings, self.embedding_fn)
            # We got multiple tokens regarding the keydowns, so we should iterate and sample from each of them:
            keydown = []
            for entry in keydown_cosine_likelyhood:
                # Softmax to normalize them all before going to the Categorical Distribution
                # /!\ Note for report: https://stackoverflow.com/questions/17187507/why-use-softmax-as-opposed-to-standard-normalization
                entry_cosine_likelyhood = torch.softmax(torch.from_numpy(entry), dim=0).squeeze()
                dist = Categorical(entry_cosine_likelyhood)
                specific_token_word = dist.sample()

                #print(f'entry_cosine_likelyhood shape: {entry_cosine_likelyhood.shape}, specific_token_word: {specific_token_word}')

                keydown.append(specific_token_word)
            keydown = torch.from_numpy(np.array(keydown))
        else:
            # TODO: Add tensor equal to empty embedding
            # by default have zeros, but we should use the proper embedding corresponding to the empty string
            #keydown = torch.zeros(512)
            keydown = torch.full((512,), self.vocabulary.stoi[self.vocabulary.padding_char])


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


        # TODO: use itos to retrieve correct number from vocab
        print(f'ref_number: {ref_number} to ', end='')
        ref_number = self.vocabulary.itos[ref_number.item()]
        print(f'ref_number: {ref_number}')

        # Get exact keydown out of the tensors:
        keydown_str = []
        for item in keydown:
            keydown_str.append(self.vocabulary.itos[item.item()])
        keydown_str = ''.join(keydown_str)
        print(f'keydown_str: {keydown_str}')

        return action.item(), ref_number, keydown_str



    # TODO: Shapes don't have the right dimensions yet.
    # - action: [timesteps, 1]
    # - ref: [timesteps, 8]
    # - keydown: [timesteps, 512]
    def evaluate(self, action, ref, keydown,
                 screenshot, new_dom, previous_action, t5_output_tensor, utterance, task_name):
        #output = self.action_layer(state) # only take first batch as we work in 1

        # XXX
        ccnet5_output_tensor, _rgb_data, _language_input, _previous_action, _t5_output_tensor = call_ccnet5(screenshot, new_dom, previous_action, t5_output_tensor, utterance, task_name)


        # a. Action type
        # Create distribution for all action types by iterating as we need to create the binary distribution
        action_probs = []
        for ac in ccnet5_output_tensor[:, 0:1].detach().numpy():
            action_binary_prob = np.array([1 - ac[0], ac[0]])
            action_probs.append(action_binary_prob)
        action_probs = torch.from_numpy(np.stack(action_probs))

        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        action_dist_probs = dist.probs


        # b. Reference Number
        ref_output = ccnet5_output_tensor[:, 1:65]
        # Iterate through the ref output, and compute the distributions
        ref_probs = []
        for ref_o in ref_output.detach().numpy():
            # Distribution must be done with Normalized List of closest embeddings
            _tokens, ref_cosine_likelyhood = embeddings2tokens(ref_o, self.embedding_fn)
            ref_cosine_likelyhood = torch.softmax(torch.from_numpy(ref_cosine_likelyhood[0]), dim=0).squeeze()
            ref_probs.append(ref_cosine_likelyhood)
        ref_probs = torch.from_numpy(np.stack(ref_probs))
        dist = Categorical(ref_probs)
        # todo: check type and shape of ref into dist.log_probs
        ref_logprobs = dist.log_prob(ref)
        ref_dist_probs = dist.probs

        # c. Keydown Text
        keydown_output = ccnet5_output_tensor[:, 65:577]
        keydown_probs = []
        # Either we create a single distribution for the full 8x64, or one for each...
        for keydown_o in keydown_output.detach().numpy():
            _tokens, keydown_cosine_likelyhood = embeddings2tokens(keydown_o, self.embedding_fn)
            # Generate a distribution for each embedding in the sequence
            level_distributions = []
            for entry in keydown_cosine_likelyhood:
                entry_cosine_likelyhood = torch.softmax(torch.from_numpy(entry), dim=0).squeeze()
                level_distributions.append(list(entry_cosine_likelyhood))
            # The keydown probs are nested with respect to the embedding position
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
        ccnet5_output_tensor, _rgb_data, _language_input, _previous_action, t5_output_tensor = call_ccnet5(screenshot, new_dom, previous_action, t5_output_tensor, utterance, task_name)


                              #todo: For this one here below
        return action_logprobs, torch.squeeze(ccnet5_output_tensor), action_dist_probs, ref_logprobs, ref_dist_probs, keydown_logprobs, keydown_dist_probs


class VMPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, embedding_fn):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.eta = torch.autograd.Variable(torch.tensor(1.0), requires_grad=True)
        self.alpha = torch.autograd.Variable(torch.tensor(0.1), requires_grad=True)
        self.eps_eta = 0.02
        self.eps_alpha = 0.1
        self.embedding_fn = embedding_fn

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, self.embedding_fn).to(device)

        params = [
            {'params': self.policy.parameters()},
            {'params': self.eta},
            {'params': self.alpha}
        ]

        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=1e-1)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, self.embedding_fn).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
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
        #old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_refs = torch.stack(memory.refs).to(device).detach()
        old_keydowns = torch.stack(memory.keydowns).to(device).detach()
        # old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Old States
        old_rgb = torch.stack(memory.rgb_data).to(device).detach()
        old_language_input = torch.stack(memory.language_input).to(device).detach()
        old_previous_actions = torch.stack(memory.previous_actions).to(device).detach()
        old_t5_out_tensors = torch.stack(memory.t5_output_tensors).to(device).detach()


        # get old probs and old advantages
        with torch.no_grad():
            # TODO: separate them, or keep them all as one?
            _, old_state_values, old_action_dist_probs, _, old_ref_dist_probs, _, old_keydown_dist_probs = self.policy_old.evaluate(old_actions, old_refs, old_keydowns,
                                                                                                                                    old_rgb, old_language_input, old_previous_actions, old_t5_out_tensors)
            advantages = rewards - old_state_values.detach()

        # Optimize policy for K epochs:
        for i in range(self.K_epochs):
            # Evaluating sampled actions and values :
            # TODO: separate them, or keep them all as one?
            action_logprobs, state_values, action_dist_probs, ref_logprobs, ref_dist_probs, keydown_logprobs, keydown_dist_probs = self.policy.evaluate(old_actions, old_refs, old_keydowns,
                                                                                                                                                        old_rgb, old_language_input, old_previous_actions, old_t5_out_tensors)

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
            loss = 1/3 * loss_action + 1/3 * loss_ref + 1/3 * avg_keydown_loss
            print(f'LOSS = {loss.shape}, {loss}     ######################################')

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            with torch.no_grad():
                self.eta.copy_(torch.clamp(self.eta, min=1e-8))
                self.alpha.copy_(torch.clamp(self.alpha, min=1e-8))
            if i == self.K_epochs - 1:
                print(f'KL_action={torch.mean(KL_action).item()}, KL_ref={torch.mean(KL_ref).item()}, KL_keydown={torch.mean(avg_KL_keydown).item()}, self.alpha.item()')
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

from miniwob.action import ActionTypes

def main(embedding_fn):
    ############## Hyperparameters ##############
    env_name = "miniwob/click-test-2-v1"
    # creating environment
    env = gym.make(env_name)
    #state_dim = env.observation_space.shape[0]
    state_dim = 512
    action_dim = 577 # Size of target output
    ref_dim = 64 # 64 Size embedding for the reference number
    keydown_dim = 8*64 # 8x64 size embedding for the target keydown word. Or should we predict an index instead?
    render = False
    solved_reward = 230  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 500  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 50 #2400  # update policy every n timesteps
    lr = 1e-4
    betas = (0.9, 0.999)
    gamma = 0.9  # discount factor
    K_epochs = 8  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = VMPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, embedding_fn)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # Episode Data
    dom_episode = []
    action_history = []
    # Set up previous_action with default values
    previous_action = torch.zeros(577)

    # Empty ref dictionaries at start of episode:
    ref_dict, rand2ref = {}, {}

    # training loop
    for i_episode in range(1, max_episodes + 1):
        #state = env.reset()
        observation, info = env.reset()
        # TODO: transform observation into input for model
        # having empty observation for now:
        #observation_numeral = np.zeros(state_dim)
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

        # 4: Model call
        # a. T5 call
        # - Format for T5 call
        # TODO: do T5 inference in the policy act()
        t5_action, t5_ref, t5_keydown = call_T5(action_history, utterance, doms)
        # format tensor T5_output into Embeddings (flattened tensor) to feed CC-NeT5
        t5_output_tensor = torch.zeros(577)
        if t5_action == 'keydown':
            t5_output_tensor[0:1] = 1
        # t_ref = torch.from_numpy(np.array(tokenizer.truncate_pad_entry(tokenizer.tokenize_string(t5_ref), 64)))
        t_ref = torch.flatten(tokenizer.embedding_fn(torch.from_numpy(np.array(tokenizer.stoi[str(t5_ref)])))).detach()
        t_keydown = torch.flatten(tokenizer.embedding_fn(torch.from_numpy(
            np.array(tokenizer.truncate_pad_entry(tokenizer.tokenize_string(str(t5_keydown)), 8))))).detach()
        t5_output_tensor[1:65] = t_ref
        t5_output_tensor[65:577] = t_keydown

        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            #action = ppo.policy_old.act(state, memory)
            # TODO: change observation to input data for model
            action, ref, keydown = ppo.policy_old.act(memory, screenshot, doms, previous_action, t5_output_tensor, utterance, env_name)
            print(f'PPO POLICY ACTION: {action}, ref: {ref}, keydown: {keydown}')
            # TODO: get correct action space

            if action == 1:
                action_type = ActionTypes.CLICK_ELEMENT
            else:
                action_type = ActionTypes.TYPE_TEXT

                # Sample and Create Action
            action = env.action_space.sample()  # Template for the action.
            action["action_type"] = env.action_space_config.action_types.index(
                action_type
            )
            # Get the ref number by using the dictionary mapping random to old reference
            try:
                action["ref"] = str(rand2ref[str(ref)])
            except:
                print(f'REF VALUE IS NOT PRESENT IN DOM DICTIONARY: {ref}')
                print(f'Attributing default ref value...')
                action["ref"] = str('4')

            # Add text if action is type_text
            if action_type == ActionTypes.TYPE_TEXT:
                action['text'] = keydown.replace(tokenizer.padding_char, '')

            # Perform action
            #state, reward, done, _ = env.step(action)
            observation, reward, terminated, truncated, info = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(terminated)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render or running_reward > (log_interval * solved_reward) * 0.8:
                env.render()
            if terminated:
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
        #break


if __name__ == '__main__':
    # Prepare Embedding Function
    embedding_fn_path = 'embedding_weights.pth'
    embedding_fn = EmbeddingFunction(vocab_size=1592, embedding_dim=64)
    embedding_fn.load_state_dict(torch.load(embedding_fn_path))

    main(embedding_fn)
