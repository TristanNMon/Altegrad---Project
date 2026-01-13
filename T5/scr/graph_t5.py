import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from encoder import GINEEncoder

class GraphT5Model(nn.Module):
    def __init__(self, 
                 graph_hidden_dim=256, 
                 t5_model_name="t5-small"):
        super().__init__()
        
        # 1. The Chemist: Custom Graph Encoder (GINE)
        self.graph_encoder = GINEEncoder(hidden_dim=graph_hidden_dim)
        
        # 2. The Author: Pre-trained T5 Decoder
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_dim = self.t5.config.d_model  # 512 for t5-small
        
        # 3. The Bridge: Project Graph Dim -> T5 Dim
        self.projector = nn.Sequential(
            nn.Linear(graph_hidden_dim, self.t5_dim),
            nn.ReLU(),
            nn.Linear(self.t5_dim, self.t5_dim),
            nn.LayerNorm(self.t5_dim)
        )

    def forward(self, batch, labels=None):
        """
        Training step
        """
        # A. Encode Graph [Batch_Size, Graph_Dim]
        graph_emb = self.graph_encoder(batch)
        
        # B. Project to T5 Dimension [Batch_Size, T5_Dim]
        projected_emb = self.projector(graph_emb)
        
        # C. Reshape for T5 [Batch_Size, Seq_Len=1, T5_Dim]
        encoder_hidden_states = projected_emb.unsqueeze(1)
        
        # D. Forward Pass through T5 Decoder
        # FIX: Pass tuple (hidden_states,) to 'encoder_outputs'
        outputs = self.t5(
            encoder_outputs=(encoder_hidden_states,), 
            labels=labels
        )
        
        return outputs

    @torch.no_grad()
    def generate_caption(self, batch, tokenizer, max_length=128, num_beams=5):
        """
        Inference step
        """
        # 1. Encode Graph
        graph_emb = self.graph_encoder(batch)
        projected_emb = self.projector(graph_emb).unsqueeze(1)
        
        # 2. Generate
        # FIX: We create a ModelOutput object or tuple for T5 generation
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=projected_emb)

        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3
        )
        
        return outputs