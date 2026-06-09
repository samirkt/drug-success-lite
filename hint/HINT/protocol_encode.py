'''
input:
	data/raw_data.csv

output:
	data/sentence2embedding.pkl (preprocessing)
	protocol_embedding 
'''

import csv, pickle 
from functools import reduce
from tqdm import tqdm 
import torch 
torch.manual_seed(0)
from torch import nn 
import torch.nn.functional as F

def clean_protocol(protocol):
	protocol = protocol.lower()
	protocol_split = protocol.split('\n')
	filter_out_empty_fn = lambda x: len(x.strip())>0
	strip_fn = lambda x:x.strip()
	protocol_split = list(filter(filter_out_empty_fn, protocol_split))	
	protocol_split = list(map(strip_fn, protocol_split))
	return protocol_split 

def get_all_protocols():
	input_file = 'data/raw_data.csv'
	with open(input_file, 'r') as csvfile:
		rows = list(csv.reader(csvfile, delimiter = ','))[1:]
	protocols = [row[9] for row in rows]
	return protocols

def split_protocol(protocol):
	protocol_split = clean_protocol(protocol)
	inclusion_idx, exclusion_idx = len(protocol_split), len(protocol_split)	
	for idx, sentence in enumerate(protocol_split):
		if "inclusion" in sentence:
			inclusion_idx = idx
			break
	for idx, sentence in enumerate(protocol_split):
		if "exclusion" in sentence:
			exclusion_idx = idx 
			break 		
	if inclusion_idx + 1 < exclusion_idx + 1 < len(protocol_split):
		inclusion_criteria = protocol_split[inclusion_idx:exclusion_idx]
		exclusion_criteria = protocol_split[exclusion_idx:]
		if not (len(inclusion_criteria) > 0 and len(exclusion_criteria) > 0):
			print(len(inclusion_criteria), len(exclusion_criteria), len(protocol_split))
			exit()
		return inclusion_criteria, exclusion_criteria ## list, list 
	else:
		return protocol_split, 

def collect_cleaned_sentence_set():
	protocol_lst = get_all_protocols() 
	cleaned_sentence_lst = []
	for protocol in protocol_lst:
		result = split_protocol(protocol)
		cleaned_sentence_lst.extend(result[0])
		if len(result)==2:
			cleaned_sentence_lst.extend(result[1])
	return set(cleaned_sentence_lst)


# PubMedBERT: a 768-d biomedical BERT with a fast tokenizer + safetensors (loads
# cleanly on modern transformers; the original BioBERT ships only a slow tokenizer).
# The protocol encoder just learns a Linear on the mean-pooled vectors, so any
# consistent biomedical sentence embedding serves the same purpose.
_BIOBERT_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
_biobert_cache = {}


def _pick_device():
	if torch.cuda.is_available():
		return "cuda"
	if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def _get_biobert(device):
	## BioBERT via transformers (the original `biobert_embedding` package is
	## unmaintained). Any consistent 768-d biomedical sentence embedding works:
	## Protocol_Embedding just learns a Linear on top of the mean-pooled vectors.
	if "model" not in _biobert_cache:
		from transformers import AutoModel, AutoTokenizer
		_biobert_cache["tok"] = AutoTokenizer.from_pretrained(_BIOBERT_NAME)
		_biobert_cache["model"] = AutoModel.from_pretrained(_BIOBERT_NAME).to(device).eval()
	return _biobert_cache["tok"], _biobert_cache["model"]


def embed_sentence_set(sentence_set, device=None, batch_size=128, max_length=128):
	"""Map each cleaned sentence -> a 768-d BioBERT mean-pooled vector (CPU float32)."""
	device = device or _pick_device()
	tok, model = _get_biobert(device)
	sentences = list(sentence_set)
	out = {}
	for i in tqdm(range(0, len(sentences), batch_size)):
		batch = sentences[i:i + batch_size]
		enc = tok(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
		enc = {k: v.to(device) for k, v in enc.items()}
		with torch.no_grad():
			hidden = model(**enc).last_hidden_state            # (B, L, 768)
		mask = enc["attention_mask"].unsqueeze(-1).float()     # (B, L, 1)
		vecs = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean-pool real tokens
		vecs = vecs.to("cpu").float()
		for s, v in zip(batch, vecs):
			out[s] = v.clone()
	return out


def save_sentence_bert_dict_pkl(sentence_set=None, device=None):
	## sentence_set defaults to the full raw_data.csv corpus; pass an explicit set
	## (e.g. just the benchmark phase sentences) to build a smaller, sufficient dict.
	if sentence_set is None:
		sentence_set = collect_cleaned_sentence_set()
	protocol_sentence_2_embedding = embed_sentence_set(sentence_set, device=device)
	pickle.dump(protocol_sentence_2_embedding, open('data/sentence2embedding.pkl', 'wb'))
	return protocol_sentence_2_embedding

def load_sentence_2_vec():
	## Path is overridable via $HINT_SENTENCE2VEC so the criteria-less path can
	## point at the tiny stub (empty dict) and skip loading the 1.1 GB real embedding.
	import os
	path = os.environ.get("HINT_SENTENCE2VEC", "data/sentence2embedding.pkl")
	sentence_2_vec = pickle.load(open(path, 'rb'))
	return sentence_2_vec

def protocol2feature(protocol, sentence_2_vec):
	result = split_protocol(protocol)
	inclusion_criteria, exclusion_criteria = result[0], result[-1]
	inclusion_feature = [sentence_2_vec[sentence].view(1,-1) for sentence in inclusion_criteria if sentence in sentence_2_vec]
	exclusion_feature = [sentence_2_vec[sentence].view(1,-1) for sentence in exclusion_criteria if sentence in sentence_2_vec]
	if inclusion_feature == []:
		inclusion_feature = torch.zeros(1,768)
	else:
		inclusion_feature = torch.cat(inclusion_feature, 0)
	if exclusion_feature == []:
		exclusion_feature = torch.zeros(1,768)
	else:
		exclusion_feature = torch.cat(exclusion_feature, 0)
	return inclusion_feature, exclusion_feature 


class Protocol_Embedding(nn.Sequential):
	def __init__(self, output_dim, highway_num, device ):
		super(Protocol_Embedding, self).__init__()	
		self.input_dim = 768  
		self.output_dim = output_dim 
		self.highway_num = highway_num 
		self.fc = nn.Linear(self.input_dim*2, output_dim)
		self.f = F.relu
		self.device = device 
		self = self.to(device)

	def forward_single(self, inclusion_feature, exclusion_feature):
		## inclusion_feature, exclusion_feature: xxx,768 
		inclusion_feature = inclusion_feature.to(self.device)
		exclusion_feature = exclusion_feature.to(self.device)
		inclusion_vec = torch.mean(inclusion_feature, 0)
		inclusion_vec = inclusion_vec.view(1,-1)
		exclusion_vec = torch.mean(exclusion_feature, 0)
		exclusion_vec = exclusion_vec.view(1,-1)
		return inclusion_vec, exclusion_vec 

	def forward(self, in_ex_feature):
		result = [self.forward_single(in_mat, ex_mat) for in_mat, ex_mat in in_ex_feature]
		inclusion_mat = [in_vec for in_vec, ex_vec in result]
		inclusion_mat = torch.cat(inclusion_mat, 0)  #### 32,768
		exclusion_mat = [ex_vec for in_vec, ex_vec in result]
		exclusion_mat = torch.cat(exclusion_mat, 0)  #### 32,768 
		protocol_mat = torch.cat([inclusion_mat, exclusion_mat], 1)
		output = self.f(self.fc(protocol_mat))
		return output 

	@property
	def embedding_size(self):
		return self.output_dim 



if __name__ == "__main__":
	# protocols = get_all_protocols()
	# split_protocols(protocols)
	save_sentence_bert_dict_pkl() 











