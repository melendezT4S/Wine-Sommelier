from langchain.vectorstores import Pinecone
import pinecone
from langchain.schema import Document
import os, time
import pandas as pd
from tqdm import tqdm
from langchain.embeddings import OpenAIEmbeddings
import openai

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

os.environ["PINECONE_CLIENT_API"] = "45a72850-7d39-48be-8c7f-aee502cc2f9c"
pinecone.init(
	api_key=os.getenv('PINECONE_CLIENT_API'),
	environment='gcp-starter'
)
embeddings = OpenAIEmbeddings()
db = Pinecone.from_existing_index(index_name='wine-db', embedding=embeddings)

# Obtiene dataframe con pandas
df = pd.read_csv('./winemag-data-reduced.csv')


# Rellena valores nulos con cadenas vacias
df = df.fillna('')

# Obtencion de muestra aleatoria 100 filas
df = df.sample(100)


# Funcion que crea  langchain.Documents a partir de una tupla
def create_document_from_tuple(t):
    return Document(
        page_content=t.description,
        metadata={
            'country': t.country,
            'province': t.province,
            'name': t.title,
            'variety':t.variety,
            'winery':t.winery
        }
    )

# Iteracion sobre el dataframe: 
# Creacion de lista de documentos langchain.Documents
# para cada fila del dataframe 
# Nota: tqdm -> barra de progreso
docs = [create_document_from_tuple(row) for row in tqdm(df.itertuples(index=False))]

# Subida de documentos a la vector store de Pinecone
print('Uploading to vector db')
s = time.perf_counter()
db.add_documents(docs)
elapsed = time.perf_counter() - s
print("\033[1m" + f"Upload executed in {elapsed:0.2f} seconds." + "\033[0m")
