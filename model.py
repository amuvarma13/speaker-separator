import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM, Wav2Vec2Model, Wav2Vec2Config


class SpeakerSeparator(Qwen3ForCausalLM):
    def __init__(self, config, w2v_name="facebook/wav2vec2-base", down=4):
        super().__init__(config)
        self.down = down
        w2v_cfg = Wav2Vec2Config.from_pretrained(w2v_name)
        w2v_cfg.gradient_checkpointing = False
        self.w2v = Wav2Vec2Model(w2v_cfg)
        self.proj = nn.Sequential(
            nn.Linear(w2v_cfg.hidden_size * down, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def encode_audio(self, audio):
        h = self.w2v(audio).last_hidden_state
        b, t, d = h.shape
        t2 = (t // self.down) * self.down
        return self.proj(h[:, :t2].reshape(b, t2 // self.down, d * self.down))

    def forward(self, audio, input_ids, latent_mask, labels=None, attention_mask=None):
        a = self.encode_audio(audio)
        embeds = self.get_input_embeddings()(input_ids)
        flat = []
        for b in range(embeds.size(0)):
            n = int(latent_mask[b].sum())
            if a.size(1) >= n:
                flat.append(a[b, :n])
            else:
                flat.append(torch.cat([a[b], a.new_zeros(n - a.size(1), a.size(2))], 0))
        flat = torch.cat(flat, 0).to(embeds.dtype)
        embeds = embeds.masked_scatter(latent_mask.unsqueeze(-1), flat)
        return super().forward(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
