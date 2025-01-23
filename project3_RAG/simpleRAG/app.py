import pathlib

import streamlit as st
import os, os.path
import shutil
import RAG

'''
pip install streamlit
streamlit run app.py
pip install pypdf
'''

## ====== create streamlit application =====
st.title("Simple RAG Demo")
st.header("\nSimple examples with Wikipedia or a pdf file")

#https://docs.streamlit.io/develop/concepts/architecture/session-state#initialization
if 'selection' not in st.session_state:
    st.session_state['selection'] = ""

resource_folder = pathlib.Path().absolute() / "docs/"
if resource_folder.exists():
    shutil.rmtree(resource_folder)
resource_folder.mkdir(parents=True, exist_ok=True)

options = False
emp = st.empty()

vari = emp.selectbox(
    key = "Options",
    label = "Please select the option for query running:",
    options = ("Wikipedia", "Research Paper", "My Scripts")
)

if st.button("Select"):
    choice = vari
    options = True

def wiki_choice():
    wiki_query = st.text_input(
        label="please input your Wikipedia search query: who is the Thanos in SquidGame2?",
        max_chars=256
    )
    if st.button("Submit"):
        return wiki_query

def research_choice():
    with st.form(key="doc_upload", clear_on_submit=False):
        uploaded_doc = st.file_uploader(
            label = "Please upload your document",
            accept_multiple_files=False,
            type=['pdf']
        )
        research_query = st.text_input(
            label = "Please input what you want to search",
            max_chars=256
        )

        submit_btn1=st.form_submit_button("Load Document")

        if submit_btn1:
            with open(os.path.join(resource_folder, uploaded_doc.name), 'wb') as f:
                f.write(uploaded_doc.getbuffer())
            return research_query

def main(selection):
    if selection == "Wikipedia":
        wiki_query = wiki_choice()

        if wiki_query is not None:
            with st.spinner("Processing your request..."):
                response = RAG.generate_response(selection, wiki_query)
                st.success("Data processing Complete!")
                st.markdown(f"###{response['result']}")

    elif selection == "Research Paper":
        research_query = research_choice()

        if research_query is not None:
            with st.spinner("Processing your request..."):
                response = RAG.generate_response(selection, research_query)
                st.success("Data processing Complete!")
                st.write(response['result'])


if __name__ == "__main__":
    if options:
        st.session_state.selection = choice
    main(st.session_state.selection)
