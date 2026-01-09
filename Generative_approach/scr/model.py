import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from encoder import GINEEncoder

class Graph2TextModel(nn.Module):
    def __init__(self, 
                 encoder_hidden_dim=256, 
                 llm_name="distilgpt2",
                 drop_ratio=0.1):
        super().__init__()
        
        # 1. The Chemist
        self.encoder = GINEEncoder(hidden_dim=encoder_hidden_dim, drop_ratio=drop_ratio)
        
        # 2. The Author
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm_dim = self.llm.config.n_embd
        
        # 3. The Adapter (FIXED)
        # We added LayerNorm to match GPT-2's internal scaling
        self.adapter = nn.Sequential(
            nn.Linear(encoder_hidden_dim, self.llm_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.LayerNorm(self.llm_dim)  # <--- CRITICAL FIX: Normalization
        )

    def forward(self, batch, input_ids=None, attention_mask=None):
        """
        Training Forward Pass
        """
        graph_emb = self.encoder(batch)
        projected_emb = self.adapter(graph_emb).unsqueeze(1) # [batch, 1, dim]
        
        if input_ids is not None:
            token_embeds = self.llm.transformer.wte(input_ids)
            
            # Concatenate: [Graph, Text]
            inputs_embeds = torch.cat((projected_emb, token_embeds), dim=1)
            
            # Masking logic
            graph_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
            extended_attention_mask = torch.cat((graph_mask, attention_mask), dim=1)
            
            # Labels logic (Ignore graph token for loss)
            ignore_index = -100
            graph_labels = torch.full((input_ids.size(0), 1), ignore_index, device=input_ids.device, dtype=torch.long)
            labels = input_ids.clone()
            labels[attention_mask == 0] = ignore_index
            extended_labels = torch.cat((graph_labels, labels), dim=1)

            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                labels=extended_labels
            )
        return projected_emb

    @torch.no_grad()
    def generate_caption(self, batch, tokenizer, max_length=128, num_beams=5):
        """
        Inference with 'Kickstarter' Token
        """
        # 1. Get Graph Vector
        graph_emb = self.encoder(batch)
        projected_emb = self.adapter(graph_emb).unsqueeze(1) # [Batch, 1, Dim]
        
        # 2. Get BOS Token Embedding (The "Kickstarter")
        # GPT-2 uses EOS as BOS usually. We fetch its embedding vector.
        # Shape: [1, 1, Dim] -> expanded to [Batch, 1, Dim]
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        bos_embed = self.llm.transformer.wte(torch.tensor([bos_token_id], device=batch.x.device))
        bos_embed = bos_embed.unsqueeze(0).expand(projected_emb.size(0), -1, -1)
        
        # 3. Concatenate [Graph_Vector, BOS_Vector]
        # Now the model sees: "Context" -> "Start" -> ... predictions ...
        inputs_embeds = torch.cat((projected_emb, bos_embed), dim=1)
        
        # 4. Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=2.0,       # <--- Increased to force variety
            no_repeat_ngram_size=3,       # <--- Hard block on repeating phrases
            do_sample=True,               # <--- Add randomness to break loops
            top_p=0.9                     # <--- Nucleus sampling
        )
        
        return outputs