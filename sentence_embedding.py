from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
import numpy as np

repo_path = "/home/haotian/Splitted_code"

loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".cpp"],
    # exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.CPP, parser_threshold=500),
)
documents = loader.load()
repo_path_2 = "/home/haotian/book"

loader2 = GenericLoader.from_filesystem(
    repo_path_2,
    glob="**/*",
    suffixes=[".txt"],
    # exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.CPP, parser_threshold=500),
)
documents2 = loader2.load()

documents = documents + documents2
len(documents)
from langchain_text_splitters import RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.CPP, chunk_size=1000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
len(texts)
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
import numpy as np
np.array(model.encode(texts[0].page_content).shape)
txt = []
for text in texts:
    txt.append(np.array(model.encode(text.page_content)))

def select_sentenceembedding(query):
    k = 5
    query_embed = model.encode(query)
    score = []
    for t in txt:
        score.append(query_embed.T @ t )
        # score.append(np.linalg.norm(query_embed - t))
    indexed_list = sorted(enumerate(score), key=lambda x: x[1], reverse= True)
    # indexed_list = sorted(enumerate(score), key=lambda x: x[1], reverse= False)
    sorted_indices = [index for index, value in indexed_list]
    top_k_index = sorted_indices[:k]
    top_k_doc = []
    for i in top_k_index:
        top_k_doc.append(texts[i])
    top_k_doc
    r_doc = []
    for t in top_k_doc:
        r_doc.append(t.page_content)
    r_doc = r_doc[:3]
    return r_doc
