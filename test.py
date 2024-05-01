import streamlit as st
import json

def generate_json_from_input(input_text):
    # Split the input text by newline character to get individual strings
    strings = input_text.split('\n')
    
    # Create a dictionary to hold the strings
    data = {"strings": strings}
    
    # Convert the dictionary to JSON format
    json_data = json.dumps(data, indent=4)
    
    return json_data

def main():
    st.title("Input Multiple Strings and Output JSON")
    
    # Create a text area for user input
    input_text = st.text_area("Enter multiple strings (one per line)")
    
    if st.button("Generate JSON"):
        # Check if input text is not empty
        if input_text:
            # Generate JSON from input text
            json_data = generate_json_from_input(input_text)
            # Display the generated JSON
            st.code(json_data, language='json')
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
