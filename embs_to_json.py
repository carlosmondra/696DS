from bert_serving.client import BertClient
import time
import json
import numpy as np
import torch
import pickle

bc = BertClient(check_length=False)

def get_sections(section_names, sections):
  if len(sections) <= 4:
    return sections
  first_section = get_first_section(section_names, sections)
  fourth_section = get_fourth_section(section_names, sections)
  second_section = get_second_section(section_names, sections, first_section)
  third_section = get_third_section(section_names, sections, fourth_section)

  return [sections[first_section], sections[second_section], sections[third_section], sections[fourth_section]]

def find_idx(section_names, sections, criterias):
  for idx, section_name in enumerate(section_names):
    for criteria in criterias:
      if criteria in section_name and sections[idx] != ['']:
        return idx
  return None

def get_first_section(section_names, sections):
  criterias = ['intro']
  idx = find_idx(section_names, sections, criterias)
  return idx if idx else 0

def get_fourth_section(section_names, sections):
  criterias = ['conclu', 'summar']
  idx = find_idx(section_names, sections, criterias)
  return idx if idx else len(section_names) - 1

def get_second_section(section_names, sections, first_section):
  criterias = ['theoretical', 'method', 'model', 'calculation', 'theory', 'architect']
  idx = find_idx(section_names, sections, criterias)
  if idx: return idx
  if first_section + 1 < len(sections):
      return first_section + 1
  return first_section

def get_third_section(section_names, sections, fourth_section):
  criterias = ['result', 'experi', 'numeric', 'comparison', 'solution', 'discussion']
  idx = find_idx(section_names, sections, criterias)
  if idx: return idx
  if fourth_section - 1 >= 0:
      return fourth_section - 1
  return fourth_section

def get_tensor_name(idx):
  str_idx = str(idx)
  return ('0' * (5 - len(str_idx) + 1)) + str_idx + '.ten'

def add_embeddings_to_json(file_name, new_path, idx_to_bytes, idx_pos=0):
  start = time.time()
  with open(file_name) as f:
    bytes_pos = idx_to_bytes[idx_pos]
    f.seek(bytes_pos)
    for idx, paper in enumerate(f, start=idx_pos):
      paper = json.loads(paper)
      sent_embs = torch.zeros([120, 768], dtype=torch.float32)
      sections = get_sections(paper['section_names'], paper['sections'])
      for pos, section in enumerate(sections):
        try:
          encoded = bc.encode(section)
          # Set a max of 30 sentences per section
          num_of_sent = len(encoded) if len(encoded) <= 30 else 30 
          sent_embs[pos*30:pos*30+num_of_sent,0:768] = torch.tensor(encoded[0:num_of_sent,0:768])
        except:
          continue
      tensor_name = get_tensor_name(idx)
      torch.save(sent_embs, new_path + tensor_name)
      if time.time() - start > 13800:
        print("last saved tensor:", idx)
        break
  print(time.time() - start)      

add_embeddings_to_json('arxiv-release/train.txt', 'arxiv-release/train_tensor/', pickle.load(open('arxiv-release/train_idx_to_bytes.pkl', 'rb')), 58496)
