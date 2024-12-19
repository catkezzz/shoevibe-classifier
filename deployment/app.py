import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Shoevibe - Home",  # Title displayed on the browser tab
    page_icon="👟",               # Optional: Set a shoe icon for branding
    layout="centered",            # Options: 'centered' or 'wide'
    initial_sidebar_state="auto", # Sidebar state
)

# Add a centered logo using columns
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths
with col2:
    st.image("shoevibe.png", use_container_width=True)  # Auto-fit the logo within the column

# Add a centered main header and subheader
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Welcome to the Shoevibe App</h1>
        <h4>Shoevibe is an NLP-based application that analyzes customer sentiment for men's shoes on Tokopedia.</h4>
        <h4>It extracts key terms from positive and negative reviews to highlight frequent words, helping buyers and sellers make informed decisions.</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for navigation (if needed)
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Display a footer for branding or additional info
st.markdown(
    """
    <hr style='border: 1px solid #ddd;'>
    <footer style='text-align: center; font-size: 12px; color: gray;'>
        © 2024 Shoevibe. All rights reserved.
    </footer>
    """,
    unsafe_allow_html=True,
)
