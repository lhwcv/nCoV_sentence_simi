import  copy
import  json
import  random
import  cv2
import torch
import csv

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

class TextFeature():
    """
       A single set of features of text.

       Args:
           input_ids: Indices of input sequence tokens in the vocabulary.
           attention_mask: Mask to avoid performing attention on padding token indices.
               Mask values selected in ``[0, 1]``:
               Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
           token_type_ids: Segment token indices to indicate first and second portions of the inputs.
           label: Label corresponding to the input
       """

    def __init__(self, input_ids, attention_mask, token_type_ids, label, input_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label = label

    def tensorfy(self, unsqueeze = True):
        if isinstance(self.input_ids, list):
            self.input_ids = torch.LongTensor(self.input_ids)
            self.attention_mask = torch.LongTensor(self.attention_mask)
            self.token_type_ids = torch.LongTensor(self.token_type_ids)
            self.label = torch.LongTensor([self.label])
            if unsqueeze:
                self.input_ids = self.input_ids.unsqueeze(0)
                self.attention_mask = self.attention_mask.unsqueeze(0)
                self.token_type_ids= self.token_type_ids.unsqueeze(0)
                self.label = self.label.unsqueeze(0)


    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TextExampleForClassify(object):
    label_list = ['0', '1']
    label_map = {label: i for i, label in enumerate(label_list)}
    """
    A single training/test Text example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence.
        text_b: string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a : str, text_b: str, label : str, whichCatgory = None):
        self.guid   = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label
        self.whichCatgory = whichCatgory



    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


    def tonkenize_to_feature(self, tokenizer, max_seq_len = 512):
        tokens_a = tokenizer.tokenize(self.text_a)
        tokens_b = tokenizer.tokenize(self.text_b)

        if self.whichCatgory is not  None:
            tokens_cato = tokenizer.tokenize(self.whichCatgory)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3 - 2*len(tokens_cato)-2)
        else:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)

        tokens = []
        token_type_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)
        if self.whichCatgory is not  None:
            for token in tokens_cato:
                tokens.append(token)
                token_type_ids.append(0)
            tokens.append("[SEP]")
            token_type_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)
        tokens.append("[SEP]")
        token_type_ids.append(0)

        if self.whichCatgory is not  None:
            for token in tokens_cato:
                tokens.append(token)
                token_type_ids.append(1)
            tokens.append("[SEP]")
            token_type_ids.append(1)

        for token in tokens_b:
            tokens.append(token)
            token_type_ids.append(1)
        tokens.append("[SEP]")
        token_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        assert len(input_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(token_type_ids) == max_seq_len
        label_id = TextExampleForClassify.label_map[self.label]
        fea = TextFeature(
                      input_ids = input_ids,
                      attention_mask = attention_mask,
                      token_type_ids = token_type_ids,
                      label = label_id,
                      input_len = input_len)
        return fea



class ImageGroupExample(object):
    """
    Image Group  Example instance contain FilePaths
    Args:
        instance_id:
        filepaths:  N frames filepath
        from_video_or_image:  is the filepaths for product video or product image
        det_box_maybe: detection box dict  {'1.jpg':[ [xmin, ymin, xmax, ymax, label] ],
                                            '2.jpg': ...}
    """
    def __init__(self, instance_id, filepaths,
                 from_video_or_image = 'video',
                 det_box_maybe = None):
        self.instance_id   = instance_id
        self.filepaths     = filepaths
        self.from_video_or_image = from_video_or_image
        self.det_box_maybe = det_box_maybe
        assert len(filepaths) == len(self.det_box_maybe)

    def random_shuffle(self):
        inxs = list(range(0, len(self.filepaths)) )
        random.shuffle(inxs)
        filepaths = [self.filepaths[i] for i in inxs]
        self.filepaths = filepaths
        if self.det_box_maybe is not None:
            det_box_maybe = [self.det_box_maybe[i] for i in inxs]
            self.det_box_maybe = det_box_maybe
        return self

    def _crop(self, img, box):
        pass

    def read_images(self, size = (192, 256)):
        pass

class InputExample():
    pass

class InputFeature():
    pass








