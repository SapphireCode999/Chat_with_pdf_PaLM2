import streamlit as st
from dotenv import load_dotenv
import multidocs_palm as mp 
import extraction as ex
from htmlTemplates import css, bot_template, user_template, info_template



st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":page_facing_up:")
st.write(css, unsafe_allow_html=True)

    
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

    
                
def main():
    load_dotenv()
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    
    st.title(':orange[Chat with multiple PDFs] :page_facing_up: :page_facing_up:')
    user_question = st.text_input("**Ask a question about your :blue[documents]:**")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader(":orange[Your documents]")
        uploaded_files = st.file_uploader(
                ":blue[Upload your files here and click on 'Process Document']. Accepts :red[pdf files only.]",
                accept_multiple_files=True)
        
        if st.button("Process Document"):
            if uploaded_files:
                with st.spinner("Processing your document(s)"):
                    # get pdf text
                    raw_text = mp.extract_docs(uploaded_files)

                    # get the text chunks
                    text_chunks = mp.chunk_texts(raw_text)

                    # create vector store
                    vectorstore = mp.get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = mp.get_chain(vectorstore)
            
                
if __name__ == '__main__':
    main()
                



   