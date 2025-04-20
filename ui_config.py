import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
            /* Page background and font */
            body {
                background-color: #0f1117;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            /* Chat bubbles */
            .chat-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 10px;
            }

            .user-bubble, .bot-bubble {
                max-width: 80%;
                padding: 15px 20px;
                border-radius: 20px;
                margin: 10px 0;
                font-size: 16px;
                word-wrap: break-word;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }

            .user-bubble {
                background-color: #1f77f3;
                color: white;
                align-self: flex-end;
            }

            .bot-bubble {
                background-color: #2e2e2e;
                color: white;
                align-self: flex-start;
            }

            /* Radio buttons & input */
            .stRadio > label {
                font-size: 16px;
                color: #ffffff;
            }

            .stTextInput > div > input {
                background-color: #20252f;
                color: white;
                border-radius: 5px;
            }

            /* Image and audio styling */
            img {
                border-radius: 10px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }

            audio {
                width: 100%;
                margin-top: 10px;
            }

            /* Download button style */
            .stDownloadButton {
                margin-top: 10px;
            }
                
            /* Transparent Download Button */
            button[data-baseweb="button"] {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid #ccc;
                padding: 6px 12px;
                border-radius: 8px;
                color: #fff;
                transition: all 0.3s ease-in-out;
            }
            button[data-baseweb="button"]:hover {
                background-color: rgba(255, 255, 255, 0.15);
                border-color: #fff;
            }

        </style>
    """, unsafe_allow_html=True)
