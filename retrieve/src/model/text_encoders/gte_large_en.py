import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

class GTELargeEN:
    def __init__(self,
                 model_path,
                 device,
                 normalize=True):
        self.device = device
        model_path = model_path if model_path is not None else 'Alibaba-NLP/gte-large-en-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, 
                                               trust_remote_code=True, 
                                               unpad_inputs=True, 
                                               use_memory_efficient_attention=True).to(device)
        self.normalize = normalize

    @torch.no_grad()
    def embed(self, text_list):
        if len(text_list) == 0:
            return torch.zeros(0, 1024)
        
        batch_dict = self.tokenizer(
            text_list, max_length=8192, padding=True,
            truncation=True, return_tensors='pt', return_token_type_ids=True).to(self.device)

        # print("input_ids shape:", batch_dict["input_ids"].shape)
        # print("input_ids max:", batch_dict["input_ids"].max().item())
        # print("token_type_ids shape:", batch_dict["token_type_ids"].shape)
        # print("token_type_ids unique:", batch_dict["token_type_ids"].unique())
        # print("type_vocab_size:", self.model.config.type_vocab_size)
        # breakpoint()

        if "token_type_ids" not in batch_dict:
            batch_dict["token_type_ids"] = torch.zeros_like(batch_dict["input_ids"]).to(self.device)
        seq_len = batch_dict["input_ids"].shape[1]
        batch_size = batch_dict["input_ids"].shape[0]
        batch_dict["position_ids"] = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(self.device)
        
        outputs = self.model(**batch_dict).last_hidden_state
        emb = outputs[:, 0]
        
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        
        return emb.cpu()

    def __call__(self, q_text, text_entity_list, relation_list):
        q_emb = self.embed([q_text])
        entity_embs = self.embed(text_entity_list)
        relation_embs = self.embed(relation_list)
        
        return q_emb, entity_embs, relation_embs
