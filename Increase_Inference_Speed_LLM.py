from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM,pipeline, AutoTokenizer, BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pandas as pd
from tqdm import tqdm
import time
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForCausalLM

# data = pd.read_csv('/media/nsl3090-3/hdd1/hujaifa/Langchain/CSV/medical_info.csv', encoding='utf-8', index_col=False)
# print('CSV Data:::')
# print(data.head)
# exit()


# csvLoader = CSVLoader('/media/nsl3090-3/hdd1/hujaifa/Langchain/CSV/medical_info.csv', encoding='utf-8')
# documents = csvLoader.load()

# # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20, separators='\n\n\n')
# docs = text_splitter.split_documents(documents)

# modelPath = "intfloat/multilingual-e5-large"
# model_kwargs = {'device':'cuda:0'}
# encode_kwargs = {'normalize_embeddings':False}
# embeddings = HuggingFaceEmbeddings(
#   model_name = modelPath,  
#   model_kwargs = model_kwargs,
#   encode_kwargs=encode_kwargs
# )

# db = FAISS.from_documents(docs, embeddings)
# questions = [
#     "アスピリンの推奨用量は何ですか？",
#     "パラセタモールの副作用は何ですか？",
#     "メトホルミンはどのような病気に使われますか？",
#     "ダイアゼパムを服用する際の注意点は何ですか？",
#     "イブプロフェンの飲み方は？",
#     "ワルファリンの主な効果は何ですか？",
#     "シンバスタチンの副作用は何ですか？",
#     "イブプロフェンの値段はいくらですか？",
#     "オメプラゾールの使用期間に関する警告は何ですか？",
#     "ギャバペンチンの副作用は何ですか？",
#     "メトフォルミンの摂取タイミングは？",
#     "ラニチジンの使用上の注意は何ですか？",
#     "セレトリジンの主な効果は何ですか？",
#     "アムロジピンの副作用は何ですか？",
#     "フルオキセチンの服用を中断する際の注意点は何ですか？",
#     "リボスタチンの適切な摂取タイミングは？",
#     "アムロキシシリンの副作用は何ですか？",
#     "シミバスタチンの効果的な利用法は？",
#     "デキサメサゾンの適切な投与量は？",
#     "ロサルタンの主な効果は何ですか？",
# ]

# ground_truths = [
#     "1日に4〜6時間ごとに水と1錠を摂取してください。",
#     "アレルギー反応が起こることがあります。",
#     "タイプ2糖尿病の治療に使用されます。",
#     "この薬を服用する間、アルコールや運転を避けてください。",
#     "食事と一緒に1錠を摂取してください。",
#     "血栓の予防に使用されます。",
#     "関節の痛みが生じることがあります。",
#     "$15です。",
#     "医師の指示なしに長期間使用しないでください。",
#     "眠気が生じることがあります。",
#     "食事と一緒に1錠を摂取してください。",
#     "腎臓の問題がある場合は使用しないでください。",
#     "アレルギー性鼻炎の治療に使用されます。",
#     "足首の腫れが生じることがあります。",
#     "突然中止しないでください。",
#     "夜に1錠を食事と一緒に摂取してください。",
#     "下痢が生じることがあります。",
#     "医師の指示に従ってください。",
#     "処方された量を守ってください。",
#     "高血圧の治療に使用されます。",
# ]



# searchDocs = db.similarity_search(questions[0])
# searchDocs2 = db.similarity_search_with_score(questions[0])
# for i in range(len(searchDocs2)):
#   print(f'Similar Chunk No.{i+1}')
#   print(searchDocs2[i][0].page_content)
#   print('Score: ',searchDocs2[i][1])
# print('Similar Chunk:::')
# print(searchDocs[0].page_content)
# exit()

tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
model = AutoModelForCausalLM.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
                                             quantization_config=bnb_config,
                                            #  load_in_8bit=True,
                                            # torch_dtype=torch.bfloat16,
                                            # attn_implementation="flash_attention_2",
                                            device_map="auto")
exit()
# ort_model = ORTModelForCausalLM.from_pretrained('elyza/ELYZA-japanese-Llama-2-7b-fast-instruct',
#                                               export=True,
#                                               provider='CUDAExecutionProvider'
#                                               )
# model = ort_model



# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device='cuda:1')
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(
    pipeline = pipe,
    model_kwargs={"temperature": 0, "max_length": 128},
)

# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 
# {context}
# Question: {question}
# Helpful Answer:"""

template = """[INST] <>
あなたは誠実で優秀な日本人のアシスタントです。
マークダウン形式で以下のコンテキスト情報を元に質問に回答してください。
<>

{context}

{question}
[/INST]"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
# print('Result:::')
qa_chain = RetrievalQA.from_chain_type(   
  llm=llm,   
  chain_type="stuff",   
  retriever=db.as_retriever(),   
  chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
) 
time_list = []

total_generated_token = 0 
# with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
for i in tqdm(range(len(questions))):
  st = time.time()
  result = qa_chain({ "query" : questions[i] })['result']
  total_generated_token += len(tokenizer.encode(result))
  et = time.time()
  time_list.append(et-st)
  print('SL:::', i+1)
  print('Question:::', questions[i])
  print('Answer:::', result)
  print('Ground Truth:::', ground_truths[i])
  print('Time:::', et-st)
  print()
  print()
    
print(f'Total Time: {sum(time_list)} Average Time: {sum(time_list)/len(time_list)}')
print(f'Total Generated Tokens: {total_generated_token} Throughput: {total_generated_token/sum(time_list)} Token/Sec')
# result = qa_chain ({ "query" : question })
# print('Result:::') 
# print(result["result"])



# template = '''
# If you don't know the answer, just say that you don't know.
# Don't try to make up an answer.
# {context}

# Respond in the persona of {persona}

# Question: {question}
# Answer:
# '''
# prompt = PromptTemplate(
#     template=template, 
#     input_variables=[
#         'context', 
#         'persona',
#         'question',
#     ]
# )

# # Initialise RetrievalQA Chain
# chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=db.as_retriever(),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": prompt}
# )

# response = chain({"query": question, "persona": persona})

# print(response) 