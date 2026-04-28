import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM, Wav2Vec2Model, Wav2Vec2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class SpeakerSeparator(Qwen3ForCausalLM):
    config_class = Qwen3Config

    def __init__(self, config, w2v_name="facebook/wav2vec2-base", down=4):
        super().__init__(config)
        self.down = down
        self.w2v = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(w2v_name))
        d_in = self.w2v.config.hidden_size * down
        d_h = config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.GELU(),
            nn.Linear(d_h, d_h),
        )

    def encode_audio(self, audio, mask=None):
        h = self.w2v(audio, attention_mask=mask).last_hidden_state
        b, t, d = h.shape
        t2 = (t // self.down) * self.down
        h = h[:, :t2].reshape(b, t2 // self.down, d * self.down)
        return self.proj(h)

    def forward(self, audio=None, audio_mask=None, input_ids=None, attention_mask=None, labels=None):
        a = self.encode_audio(audio, audio_mask)
        t = self.get_input_embeddings()(input_ids)
        embeds = torch.cat([a, t], dim=1)
        b, ta, _ = a.shape
        am = torch.ones(b, ta, device=a.device, dtype=attention_mask.dtype)
        attn = torch.cat([am, attention_mask], dim=1)
        if labels is not None:
            pad = torch.full((b, ta), -100, device=a.device, dtype=labels.dtype)
            labels = torch.cat([pad, labels], dim=1)
        return super().forward(
            inputs_embeds=embeds,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
        )
