from sentence_transformers import SentenceTransformer
# model = SentenceTransformer(
#     'all-MiniLM-L6-v2',
#     cache_folder=r'K:\MTECH\NLP\Mini Project\SemiSagev2.0\hf_cache'
# )
model = SentenceTransformer(
    'BAAI/bge-base-en-v1.5',
    cache_folder=r'K:\MTECH\NLP\Mini Project\SemiSagev2.0\hf_cache'
)