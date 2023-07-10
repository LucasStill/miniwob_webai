import json
import numpy as np
import re

dom_fields = ['ref', 'left', 'top', 'width', 'height', 'tag', 'text', 'value', 'id', 'classes', 'bg_color',
              'fg_color']

dom_fields_further = ['ref', 'tag', 'text', 'value', 'id', 'classes']


# function to iterate through the DOM elements
def iterate_dom(element):
    focused_element = None
    keydown_text = None

    # Look if the element has the 'focused': True
    if 'focused' in element.keys():
        # print(f'found "focused" for {element["tag"]}, with value {element["focused"]}')
        focused_element = element['ref']
        if 'value' in element.keys():
            keydown_text = element['value']
            # print(f'FOUND KEYDOWN TEXT "{keydown_text}", {element}')
        else:
            keydown_text = ''

    # Now delete all classes we don't need
    # todo

    # Iterate deeper
    new_children = []
    for child in element['children']:
        if type(child) is type(None):
            continue
        found_target, found_text, found_dom = iterate_dom(child)
        if found_target is not None:
            focused_element = found_target
        if found_text is not None:
            keydown_text = found_text
        new_children.append(found_dom)

    # Filter the dom
    new_dom = {}
    for field in element.keys():
        if field in dom_fields_further:#dom_fields:

            # Process some datatypes
            if field in 'bg_color' or field in 'fg_color':
                element[field] = 'rgb(' + str(round(element[field][0], 2)) + ', ' + str(round(element[field][1], 2)) + ', ' + str(round(element[field][2], 2)) + ')'
            # Get value out of array
            elif field in ['left', 'width', 'top', 'height']:
                element[field] = round(element[field].item(), 2)

            new_dom[field] = element[field]

    new_dom['children'] = new_children

    return focused_element, keydown_text, new_dom

# Example to clean a row
#target_ref, keydown_text, new_dom = iterate_dom(state['dom'])

def dict2html(html_dict):
    html_str = ''
    html_str += '<' + html_dict['tag']

    for key in html_dict.keys():
        if key not in ['children', 'tag']:
            if html_dict[key] != '' and str(html_dict[key]) != '0.0':
                html_str += ' ' + key + '=' + str(html_dict[key])

    html_str += '>'

    for kid in html_dict['children']:
        html_str += dict2html(kid)

    html_str += '</' + html_dict['tag'] + '>'

    return html_str

def prepare_t5_input(action_history, utterance, dom):

    action_history_formatted = ''
    for entry in action_history:
        try:
            if int(entry[0]) == 0:
                entry[0] = 'click'
                entry[2] = ''
            elif int(entry[0]) == 1:
                entry[0] = 'keydown'
        except:
            ''
        action_history_formatted += '{' + str(entry[0]) + ', ' + str(entry[1]) + ', ' + entry[2] + '}'

    return action_history_formatted + utterance + str(dom)

def t5_output_2_tensor(action, ref, keydown):
    action_tensor = 0
    # prepare action
    if action == 'click':
        action_tensor = 0
    elif action == 'keydown':
        action_tensor = 1
    else:
        print(f'ERROR, UNRECOGNIZED ACTION: {action}')

    # embed ref


