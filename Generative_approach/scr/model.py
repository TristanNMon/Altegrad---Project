import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from encoder import GINEEncoder

class Graph2TextModel(nn.Module):
    def __init__(self, encoder_hidden_dim=256, llm_name="distilgpt2", drop_ratio=0.1):
        super().__init__()
        
        # 1. The Chemist (Graph Encoder)
        self.encoder = GINEEncoder(hidden_dim=encoder_hidden_dim, drop_ratio=drop_ratio)
        
        # 2. The Author (LLM + LoRA)
        base_llm = AutoModelForCausalLM.from_pretrained(llm_name)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1
        )
        self.llm = get_peft_model(base_llm, peft_config)
        self.llm_dim = base_llm.config.n_embd
        
        # 3. The Projector (Bridge with Normalization)
        self.projector = nn.Sequential(
            nn.Linear(encoder_hidden_dim, self.llm_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.LayerNorm(self.llm_dim) # Prevents mode collapse
        )

    def forward(self, batch, input_ids=None, attention_mask=None):
        # A. Encode
        graph_emb = self.encoder(batch)
        projected_emb = self.projector(graph_emb) # [Batch, 768]
        
        llm_input_emb = projected_emb.unsqueeze(1) # [Batch, 1, 768]
        
        if input_ids is not None:
            # B. Prepare for Training
            token_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat((llm_input_emb, token_embeds), dim=1)
            
            # Adjust masks
            graph_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
            extended_attention_mask = torch.cat((graph_mask, attention_mask), dim=1)
            
            # Adjust labels (ignore graph token)
            ignore_index = -100
            graph_labels = torch.full((input_ids.size(0), 1), ignore_index, device=input_ids.device, dtype=torch.long)
            labels = input_ids.clone()
            labels[attention_mask == 0] = ignore_index
            extended_labels = torch.cat((graph_labels, labels), dim=1)

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                labels=extended_labels
            )
            
            # RETURN BOTH: Generation Outputs AND Projected Graph Vector
            return outputs, projected_emb
        
        return llm_input_emb

    @torch.no_grad()
    def generate_caption(self, batch, tokenizer, max_length=128, num_beams=10):
        # 1. Encode
        graph_emb = self.encoder(batch)
        projected_emb = self.projector(graph_emb).unsqueeze(1)
        
        # 2. Add 'Start' Token (BOS)
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        bos_embed = self.llm.get_input_embeddings()(torch.tensor([bos_token_id], device=batch.x.device))
        bos_embed = bos_embed.unsqueeze(0).expand(projected_emb.size(0), -1, -1)
        
        inputs_embeds = torch.cat((projected_emb, bos_embed), dim=1)
        
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], 
            dtype=torch.long, 
            device=inputs_embeds.device
        )

        # 3. Generate (High quality params)
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            temperature=0.7,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        return outputs