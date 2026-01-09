import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from encoder import GINEEncoder  # Importing the chemist from Module 2

class Graph2TextModel(nn.Module):
    """
    The Full Pipeline: Graph Encoder -> Adapter -> Pre-trained LLM.
    """
    def __init__(self, 
                 encoder_hidden_dim=256, 
                 llm_name="distilgpt2",
                 drop_ratio=0.1):
        super().__init__()
        
        # 1. The Chemist (Graph Encoder)
        # We use the GINE architecture from Module 2
        self.encoder = GINEEncoder(hidden_dim=encoder_hidden_dim, drop_ratio=drop_ratio)
        
        # 2. The Author (Pre-trained LLM)
        # We load a model with a "Language Modeling" head
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm_dim = self.llm.config.n_embd  # e.g., 768 for DistilGPT2
        
        # 3. The Adapter (The Bridge)
        # Projects the Graph Vector into the LLM's embedding space
        self.adapter = nn.Sequential(
            nn.Linear(encoder_hidden_dim, self.llm_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(self.llm_dim, self.llm_dim)
        )

    def forward(self, batch, input_ids=None, attention_mask=None):
        """
        Forward pass for TRAINING.
        Args:
            batch: PyG Batch object containing graphs
            input_ids: Tensor [batch_size, seq_len] of token indices
            attention_mask: Tensor [batch_size, seq_len] (1 for real token, 0 for pad)
        """
        # A. Encode the Graph
        # graph_emb shape: [batch_size, encoder_hidden_dim]
        graph_emb = self.encoder(batch)
        
        # B. Project to LLM Space (The "Soft Prompt")
        # projected_emb shape: [batch_size, llm_dim]
        projected_emb = self.adapter(graph_emb)
        
        # Reshape to look like a token sequence: [batch_size, 1, llm_dim]
        projected_emb = projected_emb.unsqueeze(1)
        
        # C. Prepare Input for LLM
        if input_ids is not None:
            # 1. Embed the text tokens
            # inputs_embeds shape: [batch_size, seq_len, llm_dim]
            token_embeds = self.llm.transformer.wte(input_ids)
            
            # 2. Concatenate Graph Prompt + Text
            # We place the graph embedding *before* the text tokens
            # New shape: [batch_size, 1 + seq_len, llm_dim]
            inputs_embeds = torch.cat((projected_emb, token_embeds), dim=1)
            
            # 3. Adjust Attention Mask
            # Add a '1' to the start of the mask to account for the graph token
            # New shape: [batch_size, 1 + seq_len]
            graph_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
            extended_attention_mask = torch.cat((graph_mask, attention_mask), dim=1)
            
            # 4. Feed to LLM
            # The LLM automatically computes CrossEntropy loss if we provide 'labels'
            # We align labels by ignoring the graph token for loss calculation
            # Labels: [-100 (ignore graph), token_1, token_2, ...]
            ignore_index = -100
            graph_labels = torch.full((input_ids.size(0), 1), ignore_index, device=input_ids.device, dtype=torch.long)
            
            # Standard causal masking: we mask padding tokens in the labels too
            labels = input_ids.clone()
            labels[attention_mask == 0] = ignore_index
            extended_labels = torch.cat((graph_labels, labels), dim=1)

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                labels=extended_labels
            )
            
            return outputs # Contains .loss and .logits
        
        return projected_emb

    @torch.no_grad()
    def generate_caption(self, batch, tokenizer, max_length=128, num_beams=4):
        """
        Inference method using Beam Search.
        """
        # 1. Encode Graph
        graph_emb = self.encoder(batch)
        projected_emb = self.adapter(graph_emb).unsqueeze(1) # [Batch, 1, Dim]
        
        # 2. Generate
        # We start generation with the graph embedding as the only input
        outputs = self.llm.generate(
            inputs_embeds=projected_emb,
            max_length=max_length,
            num_beams=num_beams,          # Beam search for better quality
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2        # Prevents "molecule is a molecule..." loops
        )
        
        return outputs