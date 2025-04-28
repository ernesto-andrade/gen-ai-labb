
import streamlit as st
import config as c

def update_language():
    # Update session state when the language is changed
    st.session_state['language'] = st.session_state['selected_language']

def menu():

    # Check if language is already in session_state, else initialize it with a default value
    if 'language' not in st.session_state:
        st.session_state['language'] = "Svenska"  # Default language
    
    if 'app_version' not in st.session_state:
        st.session_state['app_version'] = c.app_version
    if 'update_date' not in st.session_state:
        st.session_state['update_date'] = c.update_date

    # Sidebar for language selection
    st.sidebar.selectbox(
        "Språk", 
        ("Svenska", "English"),
        index = ["Svenska", "English"].index(st.session_state['language']),
        key = "selected_language",  # Temporary key for selected value
        on_change = update_language,  # Trigger update when changed
        label_visibility = "collapsed"
    )

    if st.session_state['language'] == "Svenska":

        st.sidebar.markdown("### Meny")

        st.sidebar.page_link("Start.py", label="Start", icon=":material/home:")
        st.sidebar.page_link("pages/transcribe.py", label="Transkribering", icon=":material/transcribe:")

        st.sidebar.markdown("# ")

    if st.session_state['language'] == "English":

        st.sidebar.markdown("### Menu")

        st.sidebar.page_link("Start.py", label="Start", icon=":material/home:")
        st.sidebar.page_link("pages/transcribe.py", label="Transcribe", icon=":material/transcribe:")

        st.sidebar.markdown("# ")