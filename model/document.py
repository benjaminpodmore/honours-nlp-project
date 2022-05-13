import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from utils import get_word_embedding, to_cuda


class DocumentEncoder(nn.Module):
    def __init__(self, tokenizer=None, model=None, layers=None):
        super(DocumentEncoder, self).__init__()

        if layers is None:
            layers = [-4, -3, -2, -1]

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
        if model is None:
            model = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased", output_hidden_states=True)

        self.tokenizer = tokenizer
        self.model = model
        self.layers = layers

    def forward(self, doc):
        encoded = self.tokenizer.encode_plus(doc.tokens, return_tensors='pt', is_split_into_words=True)
        encoded_dict = {**encoded}
        input_ids = to_cuda(encoded_dict['input_ids'])
        token_type_ids = to_cuda(encoded_dict['token_type_ids'])
        attention_mask = to_cuda(encoded_dict['attention_mask'])
        n = input_ids.shape[1]

        passes = [(i * 256, i * 256 + 512) for i in range(int(n / 256) - 1)]
        if len(passes) == 0:
            # Documents < 512 tokens
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            # Get all hidden states
            states = outputs.hidden_states
            # Stack and sum all requested layers
            output = torch.stack([states[i].squeeze() for i in [-4, -3, -2, -1]])
            output = torch.mean(output, dim=0)

            return torch.stack([get_word_embedding(encoded, output, idx) for idx, word in enumerate(doc.tokens)])
        else:
            # Document > 512 tokens
            passes.append(((int(n / 256) - 1) * 256, n))
            with torch.no_grad():
                prev_outputs = self.model(input_ids=input_ids.squeeze(0)[0:512].unsqueeze(0),
                                          token_type_ids=token_type_ids.squeeze(0)[0:512].unsqueeze(0),
                                          attention_mask=attention_mask.squeeze(0)[0:512].unsqueeze(0))

            prev_states = prev_outputs.hidden_states
            prev_output = torch.stack([prev_states[i].squeeze() for i in [-4, -3, -2, -1]])
            prev_output = torch.mean(prev_output, dim=0)
            output = prev_output[0:256]

            for i in range(1, len(passes)):
                start_i, end_i = passes[i]
                with torch.no_grad():
                    curr_outputs = self.model(input_ids=input_ids.squeeze(0)[start_i:end_i].unsqueeze(0),
                                              token_type_ids=token_type_ids.squeeze(0)[start_i:end_i].unsqueeze(0),
                                              attention_mask=attention_mask.squeeze(0)[start_i:end_i].unsqueeze(0))

                curr_states = curr_outputs.hidden_states
                curr_output = torch.stack([curr_states[i].squeeze() for i in [-4, -3, -2, -1]])
                curr_output = torch.mean(curr_output, dim=0)

                new_output = torch.mean(torch.stack((prev_output[256:512], curr_output[0:256])), dim=0)
                output = torch.cat((output, new_output))

            start_i = n - int(n / 256) * 256
            with torch.no_grad():
                curr_outputs = self.model(input_ids=input_ids.squeeze(0)[-start_i:].unsqueeze(0),
                                          token_type_ids=token_type_ids.squeeze(0)[-start_i:].unsqueeze(0),
                                          attention_mask=attention_mask.squeeze(0)[-start_i:].unsqueeze(0))

            curr_states = curr_outputs.hidden_states
            curr_output = torch.stack([curr_states[i].squeeze() for i in [-4, -3, -2, -1]])
            curr_output = torch.mean(curr_output, dim=0)
            if curr_output.dim() == 1:
                curr_output = curr_output.unsqueeze(0)

            output = torch.cat((output, curr_output))

            return torch.stack([get_word_embedding(encoded, output, idx) for idx, word in enumerate(doc.tokens)])

