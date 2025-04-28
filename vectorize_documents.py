from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Loading the embeddings model
embeddings = HuggingFaceEmbeddings()

loader = DirectoryLoader(path='data',
                        glob='**/*.pdf',
                        loader_cls=UnstructuredFileLoader,
                        show_progress=True)

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

text_chunks = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents = text_chunks, 
                                 embedding = embeddings, 
                                 persist_directory="vector_db_dir")

print("Documents vectorised... and stored in vector_db_dir")


















