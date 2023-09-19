import transformers
import torch

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')

def encode_text(text):
  input_ids = tokenizer(text, truncation=True, padding=True, return_tensors='pt')['input_ids']
  with torch.no_grad():
    embeddings = model(input_ids).last_hidden_state.mean(dim=1)
  return embeddings
import datasets

# Load the job descriptions dataset from Hugging Face
job_descriptions_dataset = datasets.load_dataset('job_descriptions')

# Extract the job descriptions from the dataset
job_descriptions = job_descriptions_dataset['train']['description']
job_description_embeddings = []
for job_description in job_descriptions:
  job_description_embedding = encode_text(job_description)
  job_description_embeddings.append(job_description_embedding)


# Read the Resume.csv file into a Python dictionary
resume_data = {}
with open('Resume.csv', 'r') as csv_file:
  csv_reader = csv.reader(csv_file)
  
  for row in csv_reader:
    resume_data[row[0]] = row[1]


# Extract the text from each CV and encode it using DistilBERT
cv_embeddings = {}
for cv_name, cv_text in resume_data.items():
  cv_embedding = encode_text(cv_text)
  cv_embeddings[cv_name] = cv_embedding
matches = {}
for job_description, job_description_embedding in zip(job_descriptions, job_description_embeddings):
  matches[job_description] = {}
  for cv_name, cv_embedding in cv_embeddings.items():
    similarity = cosine_similarity(job_description_embedding, cv_embedding)[0][0]
    matches[job_description][cv_name] = similarity
for job_description, cv_similarities in matches.items():
  sorted_cvs = sorted(cv_similarities.items(), key=lambda x: x[1], reverse=True)
  top_5_cvs = sorted_cvs[:5]
  print(f"Top 5 CVs for '{job_description}':")
  for cv_name, similarity in top_5_cvs:
    print(f"{cv_name}: {similarity}")
