import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.generation import LogitsProcessorList, GenerationConfig
import transformers
import huggingface_hub
from peft import LoraConfig, get_peft_model
import re
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import thulac
from rouge import Rouge
import lawa
import jsonlines
import numpy as np

huggingface_hub.login("...")
thu1 = thulac.thulac(seg_only=True)
rouge_cal = Rouge()

law_text_dict = {
    "失火罪":
        "第一百一十五条　"
        "放火、决水、爆炸以及投放毒害性、放射性、传染病病原体等物质或者以其他危险方法致人重伤、死亡或者使公私财产遭受重大损失的，处十年以上有期徒刑、无期徒刑或者死刑。"
        "过失犯前款罪的，处三年以上七年以下有期徒刑；情节较轻的，处三年以下有期徒刑或者拘役。"
    ,
    "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物罪":
        "第一百二十五条　"
        "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物的，处三年以上十年以下有期徒刑；情节严重的，处十年以上有期徒刑、无期徒刑或者死刑。"
        "非法制造、买卖、运输、储存毒害性、放射性、传染病病原体等物质，危害公共安全的，依照前款的规定处罚。"
        "单位犯前两款罪的，对单位判处罚金，并对其直接负责的主管人员和其他直接责任人员，依照第一款的规定处罚。"
    ,
    "挪用资金罪":
        "第一百八十五条　"
        "商业银行、证券交易所、期货交易所、证券公司、期货经纪公司、保险公司或者其他金融机构的工作人员利用职务上的便利，挪用本单位或者客户资金的，依照本法第二百七十二条的规定定罪处罚。"
        "第二百七十二条　"
        "公司、企业或者其他单位的工作人员，利用职务上的便利，挪用本单位资金归个人使用或者借贷给他人，数额较大、超过三个月未还的，或者虽未超过三个月，但数额较大、进行营利活动的，或者进行非法活动的，处三年以下有期徒刑或者拘役；挪用本单位资金数额巨大的，处三年以上七年以下有期徒刑；数额特别巨大的，处七年以上有期徒刑。"
        "有第一款行为，在提起公诉前将挪用的资金退还的，可以从轻或者减轻处罚。其中，犯罪较轻的，可以减轻或者免除处罚。"
    ,
    "挪用公款罪":
        "第一百八十五条　"
        "国有商业银行、证券交易所、期货交易所、证券公司、期货经纪公司、保险公司或者其他国有金融机构的工作人员和国有商业银行、证券交易所、期货交易所、证券公司、期货经纪公司、保险公司或者其他国有金融机构委派到前款规定中的非国有机构从事公务的人员有前款行为的，依照本法第三百八十四条的规定定罪处罚。"
        "第二百七十二条　"
        "公司、企业或者其他单位的工作人员，利用职务上的便利，挪用本单位资金归个人使用或者借贷给他人，数额较大、超过三个月未还的，或者虽未超过三个月，但数额较大、进行营利活动的，或者进行非法活动的，处三年以下有期徒刑或者拘役；挪用本单位资金数额巨大的，处三年以上七年以下有期徒刑；数额特别巨大的，处七年以上有期徒刑。"
        "国有公司、企业或者其他国有单位中从事公务的人员和国有公司、企业或者其他国有单位委派到非国有公司、企业以及其他单位从事公务的人员有前款行为的，依照本法第三百八十四条的规定定罪处罚。"
        "第三百八十四条　"
        "国家工作人员利用职务上的便利，挪用公款归个人使用，进行非法活动的，或者挪用公款数额较大、进行营利活动的，或者挪用公款数额较大、超过三个月未还的，是挪用公款罪，处五年以下有期徒刑或者拘役；情节严重的，处五年以上有期徒刑。挪用公款数额巨大不退还的，处十年以上有期徒刑或者无期徒刑。"
        "挪用用于救灾、抢险、防汛、优抚、扶贫、移民、救济款物归个人使用的，从重处罚。"
    ,
    "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票罪":
        "第二百零五条　"
        "虚开增值税专用发票或者虚开用于骗取出口退税、抵扣税款的其他发票的，处三年以下有期徒刑或者拘役，并处二万元以上二十万元以下罚金；虚开的税款数额较大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处五万元以上五十万元以下罚金；虚开的税款数额巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处五万元以上五十万元以下罚金或者没收财产。"
        "单位犯本条规定之罪的，对单位判处罚金，并对其直接负责的主管人员和其他直接责任人员，处三年以下有期徒刑或者拘役；虚开的税款数额较大或者有其他严重情节的，处三年以上十年以下有期徒刑；虚开的税款数额巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑。"
        "虚开增值税专用发票或者虚开用于骗取出口退税、抵扣税款的其他发票，是指有为他人虚开、为自己虚开、让他人为自己虚开、介绍他人虚开行为之一的。"
        "第二百零五条之一　"
        "虚开本法第二百零五条规定以外的其他发票，情节严重的，处二年以下有期徒刑、拘役或者管制，并处罚金；情节特别严重的，处二年以上七年以下有期徒刑，并处罚金。"
        "单位犯前款罪的，对单位判处罚金，并对其直接负责的主管人员和其他直接责任人员，依照前款的规定处罚。"
    ,
    "假冒注册商标罪":
        "第二百一十三条　"
        "未经注册商标所有人许可，在同一种商品、服务上使用与其注册商标相同的商标，情节严重的，处三年以下有期徒刑，并处或者单处罚金；情节特别严重的，处三年以上十年以下有期徒刑，并处罚金。"
    ,
    "非法经营罪":
        "第二百二十五条　"
        "违反国家规定，有下列非法经营行为之一，扰乱市场秩序，情节严重的，处五年以下有期徒刑或者拘役，并处或者单处违法所得一倍以上五倍以下罚金；情节特别严重的，处五年以上有期徒刑，并处违法所得一倍以上五倍以下罚金或者没收财产："
        "（一）未经许可经营法律、行政法规规定的专营、专卖物品或者其他限制买卖的物品的；"
        "（二）买卖进出口许可证、进出口原产地证明以及其他法律、行政法规规定的经营许可证或者批准文件的；"
        "（三）未经国家有关主管部门批准非法经营证券、期货、保险业务的，或者非法从事资金支付结算业务的；"
        "（四）其他严重扰乱市场秩序的非法经营行为。"
    ,
    "过失致人死亡罪":
        "第二百三十三条　"
        "过失致人死亡的，处三年以上七年以下有期徒刑；情节较轻的，处三年以下有期徒刑。本法另有规定的，依照规定。"
    ,
    "抢劫罪":
        "第二百六十三条　"
        "以暴力、胁迫或者其他方法抢劫公私财物的，处三年以上十年以下有期徒刑，并处罚金；有下列情形之一的，处十年以上有期徒刑、无期徒刑或者死刑，并处罚金或者没收财产："
        "（一）入户抢劫的；"
        "（二）在公共交通工具上抢劫的；"
        "（三）抢劫银行或者其他金融机构的；"
        "（四）多次抢劫或者抢劫数额巨大的；"
        "（五）抢劫致人重伤、死亡的；"
        "（六）冒充军警人员抢劫的；"
        "（七）持枪抢劫的；"
        "（八）抢劫军用物资或者抢险、救灾、救济物资的。"
        "第二百六十七条　"
        "携带凶器抢夺的，依照本法第二百六十三条的规定定罪处罚。"
        "第二百六十九条　"
        "犯盗窃、诈骗、抢夺罪，为窝藏赃物、抗拒抓捕或者毁灭罪证而当场使用暴力或者以暴力相威胁的，依照本法第二百六十三条的规定定罪处罚。"
        "第二百八十九条　"
        "毁坏或者抢走公私财物的，除判令退赔外，对首要分子，依照本法第二百六十三条的规定定罪处罚。"
    ,
    "诈骗罪":
        "第二百一十条　"
        "使用欺骗手段骗取增值税专用发票或者可以用于骗取出口退税、抵扣税款的其他发票的，依照本法第二百六十六条的规定定罪处罚。"
        "第二百六十六条　"
        "诈骗公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。本法另有规定的，依照规定。"
    ,
    "非法占用农用地罪":
        "第三百四十二条　违反土地管理法规，非法占用耕地、林地等农用地，改变被占用土地用途，数量较大，造成耕地、林地等农用地大量毁坏的，处五年以下有期徒刑或者拘役，并处或者单处罚金。"
    ,
    "受贿罪":
        "第一百六十三条　"
        "公司、企业或者其他单位的工作人员，利用职务上的便利，索取他人财物或者非法收受他人财物，为他人谋取利益，数额较大的，处三年以下有期徒刑或者拘役，并处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金。"
        "公司、企业或者其他单位的工作人员在经济往来中，利用职务上的便利，违反国家规定，收受各种名义的回扣、手续费，归个人所有的，依照前款的规定处罚。"
        "国有公司、企业或者其他国有单位中从事公务的人员和国有公司、企业或者其他国有单位委派到非国有公司、企业以及其他单位从事公务的人员有前两款行为的，依照本法第三百八十五条、第三百八十六条的规定定罪处罚。"
        "第一百八十四条　"
        "银行或者其他金融机构的工作人员在金融业务活动中索取他人财物或者非法收受他人财物，为他人谋取利益的，或者违反国家规定，收受各种名义的回扣、手续费，归个人所有的，依照本法第一百六十三条的规定定罪处罚。"
        "国有金融机构工作人员和国有金融机构委派到非国有金融机构从事公务的人员有前款行为的，依照本法第三百八十五条、第三百八十六条的规定定罪处罚。"
        "第三百八十三条　"
        "对犯贪污罪的，根据情节轻重，分别依照下列规定处罚："
        "（一）贪污数额较大或者有其他较重情节的，处三年以下有期徒刑或者拘役，并处罚金。"
        "（二）贪污数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金或者没收财产。"
        "（三）贪污数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产；数额特别巨大，并使国家和人民利益遭受特别重大损失的，处无期徒刑或者死刑，并处没收财产。"
        "对多次贪污未经处理的，按照累计贪污数额处罚。"
        "犯第一款罪，在提起公诉前如实供述自己罪行、真诚悔罪、积极退赃，避免、减少损害结果的发生，有第一项规定情形的，可以从轻、减轻或者免除处罚；有第二项、第三项规定情形的，可以从轻处罚。"
        "犯第一款罪，有第三项规定情形被判处死刑缓期执行的，人民法院根据犯罪情节等情况可以同时决定在其死刑缓期执行二年期满依法减为无期徒刑后，终身监禁，不得减刑、假释。"
        "第三百八十五条　"
        "国家工作人员利用职务上的便利，索取他人财物的，或者非法收受他人财物，为他人谋取利益的，是受贿罪。"
        "国家工作人员在经济往来中，违反国家规定，收受各种名义的回扣、手续费，归个人所有的，以受贿论处。"
        "第三百八十六条　"
        "对犯受贿罪的，根据受贿所得数额及情节，依照本法第三百八十三条的规定处罚。索贿的从重处罚。"
}

class LLMCVG(nn.Module):
    def __init__(self, hidden_size=3072, device="cpu", rand_init=False, model_name="meta-llama/Llama-3.2-3B"):
        super(LLMCVG, self).__init__()
        self.lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.2,
            bias="none"
        )
        print("Model Name: ", model_name)
        print("Lora Config: ", self.lora_config)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': "<|reserved_special_token_0|>"})

        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

        self.llama = get_peft_model(model, self.lora_config)

        print("Model Structure:")
        print(self.llama)
        print("\nBase Model Structure:")
        print(self.llama.base_model)

        self.num_heads = 8
        self.node_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2)
        )

        self.ptp = nn.Linear(hidden_size, 1)

        self.mse_loss_fn = nn.MSELoss()
        self.cross_loss_fn = nn.CrossEntropyLoss()

        self.law_text_dict = law_text_dict

        self.r = nn.Parameter(torch.randn(1, hidden_size), requires_grad=True)
        
        self.normal = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        
        self.AND = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        
        self.OR = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )

        self.chain_transformation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

        self.crime_specific_transformations = nn.ModuleDict({
            crime: nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1)
            ) for crime in self.law_text_dict.keys()
        })

        self.chain_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.chain_fusion = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )

    def attend_nodes(self, nodes_embs, mask=None):
        batch_size, num_nodes, hidden_size = nodes_embs.shape
        
        attn_output, attn_weights = self.node_attention(
            query=nodes_embs,
            key=nodes_embs,
            value=nodes_embs,
            key_padding_mask=mask,
            need_weights=True
        )
        
        attended_nodes = nodes_embs + attn_output
        
        return attended_nodes, attn_weights

    def gen_emb(self, text):
        # pdb.set_trace()
        text_id = self.tokenizer(text, return_tensors="pt").to(self.device)
        return self.llama.base_model.model.model.embed_tokens(text_id["input_ids"]).mean(1)

    def chain_direct_inform(self, law):
        chain_file = open("./chain_v2/"+law+"/"+"chain.txt", "r", encoding="utf-8").readlines()
        output = {}

        all_chains = []
        for line in chain_file:
            line = line.strip()
            if len(line) < 1:
                continue
            elements = line.split(" -> ")
            if len(elements) == 3:
                all_chains.append(elements)
        
        if not all_chains:
            return output
        
        chain_embeddings = []
        for chain in all_chains:
            e0 = self.gen_emb([chain[0]])
            e1 = self.gen_emb([chain[1]])
            e2 = self.gen_emb([chain[2]])

            chain_emb = torch.cat([e0, e1, e2], dim=0).unsqueeze(0)
            chain_embeddings.append(chain_emb)
        
        if chain_embeddings:
            all_chain_embs = torch.cat(chain_embeddings, dim=0)
            
            for i, chain_emb in enumerate(all_chain_embs):
                chain_emb = chain_emb.unsqueeze(0)
                attended_chain, _ = self.attend_nodes(chain_emb)
                
                original_chain_repr = attended_chain.mean(dim=1)
                
                transformed_repr = self.chain_transformation(original_chain_repr)
                
                if law in self.crime_specific_transformations:
                    crime_transformed = self.crime_specific_transformations[law](transformed_repr)
                    
                    gate_values = self.chain_gate(crime_transformed)
                    gated_repr = gate_values * crime_transformed
                    
                    final_repr = self.chain_fusion(
                        torch.cat([original_chain_repr, gated_repr], dim=-1)
                    )
                else:
                    gate_values = self.chain_gate(transformed_repr)
                    gated_repr = gate_values * transformed_repr
                    
                    final_repr = self.chain_fusion(
                        torch.cat([original_chain_repr, gated_repr], dim=-1)
                    )
                
                output[i] = final_repr
        
        return output

    def forward(self, fact, cv, term, case_cause):
        cause = case_cause[0]
        law_text = self.law_text_dict.get(cause, "")
        if not law_text:
            raise ValueError(f"Unknown case_cause: {cause}")

        law_embs = self.chain_direct_inform(cause)
        if not law_embs:
            law_ids = self.tokenizer(law_text, return_tensors='pt').input_ids.to(self.device)
            law_text_embs = self.llama.base_model.model.model.embed_tokens(law_ids)
            law_embs = {0: law_text_embs.mean(dim=1)}
        
        law_embs = torch.cat(list(law_embs.values()), dim=0)
        
        cvg_input = [
            f.strip() + "判决理由：" + c.strip() + "判决如下：" + f"判处有期徒刑{int(t.item())}个月。<|end_of_text|>"
            for f, c, t in zip(fact, cv, term)
        ]
        cvg_ids = self.tokenizer(cvg_input, return_tensors="pt", padding=True).to(self.device)
        cvg_embs = self.llama.base_model.model.model.embed_tokens(cvg_ids["input_ids"])
        cvg_label_ids = cvg_ids["input_ids"].clone()

        B = cvg_embs.size(0)
        law_embs = law_embs.repeat(B, 1, 1)

        combined_embs = torch.cat([law_embs, cvg_embs], dim=1)

        law_len = law_embs.size(1)
        mask_law = torch.full((B, law_len), -100, device=self.device)
        cvg_label_ids = torch.cat([mask_law, cvg_label_ids], dim=-1)
        
        fact_prompt_tokens = [
            self.tokenizer(f.strip() + "判决理由：", return_tensors="pt")["input_ids"][0]
            for f in fact
        ]
        fact_lengths = [len(tokens) for tokens in fact_prompt_tokens]
        
        for i, length in enumerate(fact_lengths):
            cvg_label_ids[i, law_len:law_len+length] = -100

        attention_mask = torch.cat(
            [
                torch.ones((B, law_len), dtype=torch.long, device=self.device),
                cvg_ids["attention_mask"]
            ],
            dim=1
        )

        outputs = self.llama(
            inputs_embeds=combined_embs,
            attention_mask=attention_mask,
            labels=cvg_label_ids.long(),
            return_dict=True,
            output_hidden_states=True
        )

        logits = outputs.logits
        cvg_loss = outputs.loss

        ptp_label = [f"判处有期徒刑{int(t.item())}个月。<|end_of_text|>" for t in term]
        ptp_ids = self.tokenizer(ptp_label, return_tensors="pt", padding=True).to(self.device)["input_ids"][:, 1:]

        ptp_len = ptp_ids.size(1)
        ptp_logits = logits[:, -ptp_len-1:-1, :]

        ptp_loss = self.cross_loss_fn(
            ptp_logits.reshape(-1, self.llama.vocab_size),
            ptp_ids.reshape(-1)
        )
        
        return cvg_loss, ptp_loss

    def ske_generate(self, fact, case_cause):
        cause = case_cause[0]
        law_text = self.law_text_dict.get(cause, "")
        if not law_text:
            raise ValueError(f"Unknown case_cause: {cause}")

        law_embs = self.chain_direct_inform(cause)
        if not law_embs:
            law_ids = self.tokenizer(law_text, return_tensors='pt').input_ids.to(self.device)
            law_text_embs = self.llama.base_model.model.model.embed_tokens(law_ids)
            law_embs = {0: law_text_embs.mean(dim=1)}
        
        law_embs = torch.cat(list(law_embs.values()), dim=0)

        cvg_input = [f.strip() + "判决理由：" for f in fact]
        cvg_ids = self.tokenizer(cvg_input, return_tensors="pt", padding=True).to(self.device)
        cvg_embs = self.llama.base_model.model.model.embed_tokens(cvg_ids["input_ids"])

        B = cvg_embs.shape[0]
        law_embs = law_embs.repeat(B, 1, 1)

        combined_embs = torch.cat([law_embs, cvg_embs], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones((B, law_embs.size(1)), dtype=torch.long, device=self.device),
                cvg_ids["attention_mask"]
            ],
            dim=1
        )

        gen_config = transformers.GenerationConfig(
            max_new_tokens = 400,
            min_new_tokens = None,
            do_sample = True,
            use_cache = False,
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.eos_token_id,

            temperature = 0.6,
            top_k = 5,
            top_p = 0.1,
            typical_p = 1.0,
            repetition_penalty = 1.176,

            num_return_sequences = 1,
            output_hidden_states = True
        )

        outputs = self.llama.generate(
            inputs_embeds=combined_embs,
            attention_mask=attention_mask,
            generation_config=gen_config,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        gen_text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        pattern = r"判处有期徒刑([1-9]?[0-9]{1,2}|[1-2][0-9]{2}|300)个月。"
        ptp_out = []
        for txt in gen_text:
            match = re.search(pattern, txt)
            if match:
                ptp_out.append(int(match.group(1)))
            else:
                ptp_out.append(0)

        return gen_text, ptp_out

    def truncate_text(text, max_length):
        if len(text) <= max_length:
            return text
        return text[:max_length]

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 1

    model = LLMCVG(device=device).to(device).bfloat16()

    param_groups = [
        {'params': [p for n, p in model.named_parameters() if "lora" in n], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if "node_attention" in n or "fusion_layer" in n], 'lr': 3e-5},
        {'params': [p for n, p in model.named_parameters() if "chain_transformation" in n
                    or "crime_specific_transformations" in n
                    or "chain_gate" in n
                    or "chain_fusion" in n], 'lr': 4e-5},  
        {'params': [p for n, p in model.named_parameters() if "normal" in n or "AND" in n or "OR" in n or "r" == n], 'lr': 2e-5}
    ]
    
    optimizer = torch.optim.Adam(param_groups)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    for name, param in model.named_parameters():
        if "lora" not in name and "llama" in name:
            param.requires_grad = False
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Trainable param =>", name)

    train = open("./...json", 'r', encoding = 'utf-8').readlines()
    train_data = []
    for line in train:
        line = json.loads(line)
        if len(line["opinion"]) < 1:
            continue
        if len(line["justice"].strip()) > 512:
            original_justice_len = len(line["justice"].strip())
            line["justice"] = line["justice"].strip()[:512]
        if len(line["opinion"]) > 400:
            original_opinion_len = len(line["opinion"])
            line["opinion"] = line["opinion"][:400]
        train_data.append(line)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    test = open("./...json", 'r', encoding='utf-8').readlines()
    test_data = []
    for line in test:
        line = json.loads(line)
        if len(line["opinion"]) < 1:
            continue
        if len(line["justice"].strip()) > 512:
            line["justice"] = line["justice"].strip()[:512]
        if len(line["opinion"]) > 400:
            line["opinion"] = line["opinion"][:400]
        test_data.append(line)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    loss_fct = nn.MSELoss()
    mae = nn.L1Loss()

    best_rmse = float("inf")
    best_r2 = 0
    count = 0
    patience = 15
    
    for e in range(epochs):
        epoch_loss, epoch_loss_gen = 0, 0
        iters = 0
        model.train()
        
        for batch in tqdm(train_dataloader):
            case_cause = batch["caseCause"]
            cvg_loss, ptp_loss = model.forward(batch["justice"], batch["opinion"], batch["judge"], case_cause)
            
            total_loss = cvg_loss + ptp_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += ptp_loss.item()
            epoch_loss_gen += cvg_loss.item()
            iters += 1

        print(f"Epoch {e}: Prediction Loss: {epoch_loss / iters:.2f}, Generation Loss: {epoch_loss_gen / iters:.2f}")
        
        model.eval()
        pres = []
        target = []
        cv_reference = []
        cv_generate = []

        for batch in tqdm(test_dataloader):
            case_cause = batch["caseCause"]
            gcv, pre = model.ske_generate(batch["justice"], case_cause)

            target.extend(batch["judge"])
            pres.extend(pre)

            cv_generate.extend(gcv)
            for opinion, judge_item in zip(batch["opinion"], batch["judge"]):
                cv_reference.append(opinion + f"判决如下：判处有期徒刑{int(judge_item)}个月。")

        mae_r = mae(torch.tensor(pres).float().cpu(), torch.tensor(target))
        rmse_r = torch.sqrt(loss_fct(torch.tensor(pres).float().cpu(), torch.tensor(target))).item()

        all_sum = [" ".join(lawa.cut(pred.replace(" ", ""))) for pred in cv_reference]
        all_can = [" ".join(lawa.cut(pred.replace(" ", ""))) for pred in cv_generate]
        rougescore = []
        for ref, hyp in zip(all_sum, all_can):
            if len(hyp.split(" ")) < 2:
                continue
            try:
                _rougescore = rouge_cal.get_scores(ref, hyp, avg=True)
                rougescore.append([
                    list(_rougescore["rouge-1"].values())[-1],
                    list(_rougescore["rouge-2"].values())[-1],
                    list(_rougescore["rouge-l"].values())[-1]
                ])
            except:
                continue
        
        scheduler.step(rmse_r)

        print(f"Epoch {e}: MSE: {loss_fct(torch.tensor(pres).float().cpu(), torch.tensor(target)):.2f}")
        print(f"Epoch {e}: MAE: {mae_r:.2f}")
        print(f"Epoch {e}: RMSE: {rmse_r:.2f}")

        if len(rougescore) > 0:
            column_means = np.mean(np.array(rougescore), axis=0)
            print(f"Epoch {e}: Rouge-1: {column_means[0] * 100:.2f}, Rouge-2: {column_means[1] * 100:.2f}, Rouge-L: {column_means[2] * 100:.2f}")
            print("REF:", all_sum[0].replace(" ", ""))
            print("HYP:", all_can[0].replace(" ", ""))

            if e % 5 == 0 or rmse_r < best_rmse:
                checkpoint = {
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_rmse': best_rmse,
                    'best_r2': best_r2
                }
                torch.save(checkpoint, f"./save_models/checkpoint_epoch_{e}.pt")

            score = (best_rmse - rmse_r) + (column_means[1] * 100 - best_r2)
            if score > 0:
                best_rmse = rmse_r
                best_r2 = column_means[1] * 100
                count = 0
                best_pres = pres
                best_target = target
                best_ref = cv_reference
                best_hyp = cv_generate

                torch.save(model.state_dict(), "./save_models/"+"best_"+str(e))
            else:
                count += 1
        else:
            if rmse_r < best_rmse:
                best_rmse = rmse_r
                count = 0
                best_pres = pres
                best_target = target
                best_ref = cv_reference
                best_hyp = cv_generate

                torch.save(model.state_dict(), "./save_models/" + "best_" + str(e))
            else:
                count += 1

        if count >= patience:
            print("Early stopping because metric not improving.")
            break

    print("*" * 10 + "BEST" + "*" * 10)
    print(f"BEST Result => RMSE: {best_rmse:.2f}")

    for pre, tar, hyp, ref in zip(best_pres, best_target, best_hyp, best_ref):
        if isinstance(tar, torch.Tensor):
            tar = tar.item()
        if isinstance(pre, torch.Tensor):
            pre = pre.item()
        line = {
            "Target": tar,
            "Prediction": pre,
            "Reference": ref,
            "Generate": hyp.replace(" ", "")
        }
        with jsonlines.open("./output/...json", mode='a') as writer:
            writer.write(line)

    print("Done.")
