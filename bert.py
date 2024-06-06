import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *

import math

class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
    # Attention scores are calculated by multiplying the key and query to obtain
    # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].
    # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
    # token, given by i-th attention head.
    # Before normalizing the scores, use the attention mask to mask out the padding token scores.
    # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
    # and padding tokens (with a value of a large negative number).

    # Make sure to:
    # - Normalize the scores with softmax.
    # - Multiply the attention scores with the value to get back weighted values.
    # - Before returning, concatenate multi-heads to recover the original shape:
    #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].

    ### TODO
    # Q shape: [batch_size, num_attention_heads, seq_len, attention_head_size]
    # K shape: [batch_size, num_attention_heads, seq_len, attention_head_size]
    # V shape: [batch_size, num_attention_heads, seq_len, attention_head_size]

    ## Calculate attention scores:  QK^T / sqrt(d_k)
    d_k = key.size(-1)  # attention head size 
    key_T = key.transpose(-1, -2) # K shape: [batch_size, num_attention_heads, attention_head_size, seq_len]
    attention_scores = torch.matmul(query, key_T) / math.sqrt(d_k)
    
    ## Before normalizing the scores, use the attention mask to mask out the padding token scores.
    attention_scores += attention_mask

    ## Normalize the scores
    # attention_scores shape: [batch_size, num_attention_heads, seq_len, seq_len]
    attention_probabilities = nn.Softmax(dim=-1)(attention_scores) # apply softmax along the dim that represents different key positions for each query position 
    ### REVISIT: Could use dropout here ?? => start on a few of the attention heads 
      # thought it might be useful not to overfit to a single task 
      # our method, here's some latex 
      # this is what we tried beyond the original code and why
    
    # attention_probabilities = self.dropout(attention_probabilities)

    ## Multiply the attention scores with the value to get back weighted values
    weighted_sum_vals = torch.matmul(attention_probabilities, value)
    # dim of weighted_sum_vals: [batch_size, num_attention_heads, seq_len, attention_head_size]
    bs = key.size(0)
    seq_len = key.size(2)
    attention_layer = weighted_sum_vals.transpose(1, 2).contiguous().view(bs, seq_len, self.all_head_size)
    return attention_layer


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = BertSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    This function is applied after the multi-head attention layer or the feed forward layer.
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies dropout to the transformed output of each sub-layer,
    # before it is added to the sub-layer input and normalized with a layer norm.
    ### TODO
    dropout_output = dropout(dense_layer(output))
    output = ln_layer(input + dropout_output)
    return output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
    ### TODO
    attention_layer = self.self_attention(hidden_states, attention_mask)
    attention_output = self.add_norm(hidden_states, attention_layer, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
    feed_forward_output = self.interm_af(self.interm_dense(attention_output))
    output = self.add_norm(attention_output, feed_forward_output, self.out_dense, self.out_dropout, self.out_layer_norm)
    return output



class BertModel(BertPreTrainedModel):
  """
  The BERT model returns the final embeddings for each token in a sentence.
  
  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n BERT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # BERT encoder.
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids=None, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # If input_ids is provided, compute embeddings as usual
            input_shape = input_ids.size()
            seq_length = input_shape[1]
            inputs_embeds = self.word_embedding(input_ids)

            # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
            pos_ids = self.position_ids[:, :seq_length]
            pos_embeds = self.pos_embedding(pos_ids)

            # Get token type ids. Since we are not considering token type, this embedding is
            # just a placeholder.
            tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
            tk_type_embeds = self.tk_type_embedding(tk_type_ids)

            # Add three embeddings together; then apply embed_layer_norm and dropout
            hidden_states = inputs_embeds + pos_embeds + tk_type_embeds
            hidden_states = self.embed_layer_norm(hidden_states)
            hidden_states = self.embed_dropout(hidden_states)
            return hidden_states
        elif inputs_embeds is not None:
            # If inputs_embeds are provided, use them directly (no need to compute)
            # However, still apply layer normalization and dropout for consistency
            hidden_states = self.embed_layer_norm(inputs_embeds)
            hidden_states = self.embed_dropout(hidden_states)
            return hidden_states
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.bert_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  

  def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        inputs_embeds: [batch_size, seq_len, hidden_size] (Optional)
        """

        # If input_ids is provided, but not inputs_embeds, then use embed as normal
        if input_ids is not None and inputs_embeds is None:
            embedding_output = self.embed(input_ids=input_ids)
        elif input_ids is None and inputs_embeds is not None:
            # If inputs_embeds are provided, use them directly
            embedding_output = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Feed to a transformer (a stack of BertLayers).
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # Get cls token hidden state.
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}