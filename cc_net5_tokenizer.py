import torch
from cosine_similarity import EmbeddingFunction

css_fields = ['top', 'left', 'width', 'height']
special_characters = ['.', ',', '#', ':', '-', '/', '(', ')', 'https://', '@', '&', '"', "'", '!', '?', ';', '+', '=',
                      '*', '$', '€', '*', '`']


def round_to_nearest_ten(number):
    return round(number / 10) * 10

# Turn it into a class
class CCNeT5Tokenizer:
    def __init__(self, vocab_path):
        stoi = {}
        itos = {}
        self.padding_char = '<PAD>'
        self.special_characters = ['.', ',', '#', ':', '-', '/', '(', ')', 'https://', '@', '&', '"', "'", '!', '?',
                                   ';', '+', '=', '*', '$', '€', '*', '`']

        with open(vocab_path, 'r') as file:
            for index, line in enumerate(file):
                line = line.strip()
                stoi[line] = index
                itos[index] = line

        stoi[' '] = stoi['']
        itos[stoi[' ']] = ' '
        self.stoi = stoi
        self.stoi[self.padding_char] = len(stoi.keys())-1  # Add PADDING character
        # We do not need itos as we don't implement a de-tokinezing function.
        self.itos = itos
        self.itos[len(stoi.keys()) - 2] = self.padding_char  # Add PADDING character
        print(f'pad: {len(stoi.keys()) - 2}')
        print(f'Loaded CC_NeT5 Tokenizer with vocabulary size being {len(self.stoi)}.')

        # Instantiate the embedding function
        vocab_size = len(stoi)

    # Provide a string and tokenize it: utterance or task name.
    def tokenize_string(self, string):
        tokenized_string = []
        for w in str(string).lower().split(' '):
            for sc in self.special_characters:
                if sc in w:
                    w = w.replace(sc, 'Ø' + sc + 'Ø')
            p1_words = [s for s in w.split('Ø') if s != '']
            # Check if the found words exist in our vocab, else subdivide
            for fw in p1_words:
                if fw not in self.stoi.keys():
                    for c in fw:
                        tokenized_string.append(self.stoi[c])
                else:
                    tokenized_string.append(self.stoi[fw])
        return tokenized_string

    # Turns an array into a string
    def detokenize_array(self, array):
        reconstruted_string = []
        #pad_index = self.itos[self.stoi[self.padding_char]]
        for v in array:
            v = int(v)
            s = self.itos[v]
            if s != self.padding_char:
                reconstruted_string.append(s)

        #reconstruted_string = ' '.join(reconstruted_string)
        cleaned_string = ''
        last_single = False
        for s in reconstruted_string:
            if len(s) > 1:
                if last_single:
                    cleaned_string += ' '
                    last_single = False
                cleaned_string += s + ' '
            else:
                # We do this to treat single length characters
                cleaned_string += s
                last_single = True
        return cleaned_string

    # Truncate and pad the incoming sentences based on a max_size argument
    def truncate_pad_entry(self, tokenized_array, max_size):
        if len(tokenized_array) < max_size:
            while len(tokenized_array) < max_size:
                tokenized_array.append(self.stoi[self.padding_char])
            return tokenized_array
        elif len(tokenized_array) > max_size:
            # Too long, so we automatically truncate
            return tokenized_array[:max_size]
        else:
            # Just avoid iterating under the hood
            return tokenized_array

    # Tokenize a DOM dictionary
    def tokenize_dom(self, dom):
        tokenized_dom = []

        # Add opening tag
        element_tag = dom['tag'].lower()
        if 'input' in dom.keys():
            # if tag is input, reformat it
            element_tag = dom['input'].lower() + '_' + dom['type'].lower()
            tokenized_dom.append(self.stoi['<' + element_tag])
        else:
            # add normal tag
            try:
                tokenized_dom.append(self.stoi['<' + dom['tag'].lower()])
            except:
                tokenized_dom.append(self.stoi['<'])
                tokenized_dom.append(self.stoi[dom['tag'].lower()])

        for field in dom:
            if field != 'tag' and field != 'children' and field != 'type' and field != 'text' and field not in css_fields:  # and field != 'value':

                tokenized_dom.append(self.stoi[field.lower()])

                # Ensure we don't have float values, we'll stick with integers
                if isinstance(dom[field], float):
                    tokenized_dom.append(self.stoi[str(int(round_to_nearest_ten(dom[field])))])
                else:
                    words = str(dom[field]).lower().split(' ')
                    for word in words:
                        for sc in special_characters:
                            if sc in word:
                                word = word.replace(sc, 'Ø' + sc + 'Ø')
                        p1_words = [s for s in word.split('Ø') if s != '']

                        # decides whether we keep the full word, or just the letters:
                        if field == 'value' or (False and (field == 'label' or field == 'button')):
                            for w in p1_words:
                                # Take individual characters of the value string
                                processed_words = [self.stoi[w[i:i + 1]] for i in range(0, len(w), 1)]
                                tokenized_dom += processed_words

                        elif field == 'ref':
                            for w in p1_words:
                                # Take individual characters of the value string
                                processed_words = [self.stoi[w[i:i + 3]] for i in range(0, len(w), 3)]
                                tokenized_dom += processed_words
                        else:
                            # Use the full word:
                            for w in p1_words:
                                try:
                                    tokenized_dom.append(self.stoi[str(w)])
                                except:
                                    for s in w:
                                        try:
                                            tokenized_dom.append(self.stoi[str(s)])
                                        except:
                                            ''

            elif field == 'text':
                tokenized_dom.append(self.stoi['text'])
                for c in dom[field]:
                    tokenized_dom.append(self.stoi[c])
            elif field in css_fields:
                # Cast field
                tokenized_dom.append(self.stoi[field])
                css_value = int(round_to_nearest_ten(float(dom[field])))
                tokenized_dom.append(self.stoi[str(css_value)])

        if 'children' in dom.keys():
            tokenized_dom.append(self.stoi['children'])
            for child in dom['children']:
                tokenized_dom += self.tokenize_dom(child)
                # for v in found_vocab:
                #    tokenized_dom.append(self.stoi[v])

        # add closing tag
        tokenized_dom.append(self.stoi['</' + element_tag + '>'])

        return tokenized_dom