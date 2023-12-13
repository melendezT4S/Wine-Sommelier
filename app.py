from llama_index import (
    StorageContext,
    load_index_from_storage,
)
import os
import openai

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import langchain
import os
import streamlit as st

from chains import build_query_chain, build_recommendation_chain

#Openai and LLM setup
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

langchain.debug = True
llm = OpenAI(temperature=0.0)

#Get index with embeddings
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

#Get retriever for semantic search
retriever = index.as_retriever() #Returns a Vector Store Retriever
retriever.similarity_top_k = 3 #Will retrieve 3 Nodes on List[NodeWithScore] when retriever.retrieve

def search_wines(query, retriever):
    #Return nodes  most similar to query, along with scores.
    docs = retriever.retrieve(query)
    return docs

def extract_fields(nodes):
    diccionario_general = {}
    i=0
    for node in nodes:
        lineas = node.text.split('\n')

        # Inicializar variables para almacenar los campos y sus valores
        campos = {}
    
        # Iterar sobre cada línea para extraer los campos y valores
        for linea in lineas:
            if ':' in linea:
                campo, valor = linea.split(':', 1)
                campos[campo.strip()] = valor.strip()
        diccionario_general[f"TextNode_{i}"]= campos
        i = i + 1
    return diccionario_general



def main():
    st.title("Wine Sommelier")

    st.write("Fill in the form in the sidebar to get a wine recommendation")

    # Ask the questions and store answers
    st.sidebar.subheader("1. Preferred Taste Profile")
    taste = st.sidebar.selectbox(
        "", ["Select an option", "Very sweet", "Sweet", "Neutral", "Dry", "Very dry"]
    )

    st.sidebar.subheader("2. Wine Experience")
    experience = st.sidebar.selectbox(
        "",
        [
            "Select an option",
            "Novice (just starting)",
            "Casual drinker (have tried a few)",
            "Enthusiast (drink regularly)",
            "Connoisseur (very knowledgeable)",
        ],
    )

    st.sidebar.subheader("3. Red vs. White")
    wine_color = st.sidebar.selectbox(
        "",
        [
            "Select an option",
            "Prefer red",
            "Prefer white",
            "Like both equally",
            "No preference",
        ],
    )

    st.sidebar.subheader("4. Favorite Flavors")
    flavors = [
        "Fruity",
        "Earthy",
        "Floral",
        "Spicy",
        "Oaky/Woody",
        "Citrusy",
        "Buttery",
        "Herbal",
    ]
    flavor = st.sidebar.multiselect("", flavors)

    st.sidebar.subheader("5. Pairing Intent")
    pairing = st.sidebar.selectbox(
        "",
        [
            "Select an option",
            "Casual drinking",
            "Romantic dinner",
            "Seafood meal",
            "Red meat dish",
            "Poultry dish",
            "Vegetarian meal",
            "Dessert",
            "No specific pairing",
        ],
    )

    st.sidebar.subheader(
        "6. Any complement of the answers above with other foods or tastes you like the most."
    )
    complement = st.sidebar.text_input("Complement the answers above")

    query_chain, query_response_format, query_output_parser = build_query_chain(llm)
    
    (
        recommend_chain,
        recommend_response_format,
        recommend_output_parser,
    ) = build_recommendation_chain(llm)

    if st.checkbox("Generate recommendation"):
        response = query_chain.run(
            {
                "taste": taste,
                "experience": experience,
                "wine_color": wine_color,
                "flavor": flavor,
                "pairing": pairing,
                "complement": complement,
                "response_format": query_response_format,
            }
        )

        query = query_output_parser.parse(response)["query_string"]
        print(query)
        docs = search_wines(query=query, retriever=retriever)
        dicts = extract_fields(docs)
        wine_options = [
            {
                "name": dicts[dic_key]["title"],
                "country": dicts[dic_key]["country"],
                "province": dicts[dic_key]["province"],
                "variety": dicts[dic_key]["variety"],
                "winery": dicts[dic_key]["winery"],
            }
            for dic_key in dicts
        ]

        wine_1 = wine_options[0]
        wine_2 = wine_options[1]
        wine_3 = wine_options[2]

        response = recommend_chain.run(
            {
                "taste": taste,
                "experience": experience,
                "wine_color": wine_color,
                "flavor": flavor,
                "pairing": pairing,
                "complement": complement,
                "response_format": recommend_response_format,
                "wine_1": wine_1,
                "wine_2": wine_2,
                "wine_3": wine_3,
            }
        )

        recommendation = recommend_output_parser.parse(response)["recommendation"]
        explanation = recommend_output_parser.parse(response)["explanation"]

        matching_wine = next(
            (wine for wine in wine_options if wine["name"] == recommendation), None
        )

        st.write("---")
        try:
            st.write(
                f"""
                    ### 🍷 **Recommended Wine**
                    {recommendation}

                    ---
                    **Country**: {matching_wine['country']}

                    **Province**: {matching_wine['province']}

                    **Variety**: {matching_wine['variety']}

                    **Winery**: {matching_wine['winery']}

                    **Explanation**: {explanation}
                    """
            )
        except:
            st.write(
                f"""
                    ### 🍷 **Recommended Wine**
                    {recommendation}

                    **Explanation**: {explanation}
                    """
            )


if __name__ == "__main__":
    main()
