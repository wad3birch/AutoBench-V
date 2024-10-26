import streamlit as st
import sys
import json
import os
from PIL import Image, ImageDraw, ImageFilter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.lvm_pool import gpt4o

def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def translate_text(text):
    return gpt4o(f"translate English to Chinese: {text}")

def add_corners(im, rad):
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im

# Set up the Streamlit webpage
st.set_page_config(page_title="Human Evaluation Platform", layout="wide")
st.title("Human Evaluation Platform")

logo_path = "config/logo.jpg"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    logo = logo.convert("RGBA")
    logo = add_corners(logo, 30)
    logo = logo.filter(ImageFilter.SMOOTH)

    st.sidebar.image(logo, width=200)

st.sidebar.header("Settings")
difficulty = st.sidebar.selectbox("Select Difficulty:", ["easy", "medium", "hard"])
eval_type = st.sidebar.selectbox("Select Eval Type:", ["aligned_image_json", "questions"])
user_input = st.sidebar.selectbox("User Input:", [
    "atmospheric_understanding", "basic_understanding", "reasoning_capacity",
    "semantic_understanding", "spatial_understanding"
])

if user_input and difficulty and eval_type:
    save_file_path = f'./document/{user_input}/human_eval/{eval_type}/{difficulty}_user_choices.json'
    if not os.path.exists(f'./document/{user_input}/human_eval/{eval_type}'):
        os.makedirs(f'./document/{user_input}/human_eval/{eval_type}')
    if len(load_data(save_file_path)) == 0:
        data_file_path = f'./document/{user_input}/{eval_type}/{difficulty}_{eval_type}.json'
        if eval_type == "aligned_image_json":
            data_file_path = f'./document/{user_input}/{eval_type}/{difficulty}_aligned_images.json'
    else:
        data_file_path = save_file_path

    data = load_data(data_file_path)
    st.sidebar.write(f"Loaded {len(data)} items.")

    current_index = st.sidebar.number_input("Image Index", min_value=0, max_value=len(data)-1, step=1)
    item = data[current_index]

    # Display the image and the text side by side
    col1, col2 = st.columns(2)

    with col1:
        # Display the image
        image_path = item["image_path"]
        image = Image.open(image_path)
        image = image.resize((300, 225), Image.Resampling.LANCZOS)  # Resize image to 300x225
        st.image(image, caption=f"Image {current_index + 1} / {len(data)}", use_column_width=True)
        
        # Auto mark status based on user_choice
        is_marked = 'user_choice' in item
        if is_marked:
            if item['user_choice'] == "align":
                mark_status = '<p style="color: green; font-weight: bold;">Marked: Align</p>'
            else:
                mark_status = '<p style="color: red; font-weight: bold;">Marked: Not Align</p>'
        else:
            mark_status = '<p style="color: gray; font-weight: bold;">Not Marked</p>'
        item['marked'] = is_marked
        save_data(data, save_file_path)

        st.markdown(mark_status, unsafe_allow_html=True)

    with col2:
        # Display and translate prompt, question, and answer
        prompt = item['prompt']
        question = item.get('objective_question', 'N/A')
        answer = item.get('objective_reference_answer', 'N/A')

        st.subheader("Prompt")
        if st.button("Translate Prompt"):
            prompt = translate_text(prompt)
            item['prompt'] = prompt
            save_data(data, save_file_path)  # Save the translated text back to the file
        st.write(prompt)

        st.subheader("Question")
        if question != 'N/A' and st.button("Translate Question"):
            question = translate_text(question)
            item['objective_question'] = question
            save_data(data, save_file_path)  # Save the translated text back to the file
        st.write(question)

        st.subheader("Answer")
        if answer != 'N/A' and st.button("Translate Answer"):
            answer = translate_text(answer)
            item['objective_reference_answer'] = answer
            save_data(data, save_file_path)  # Save the translated text back to the file
        st.write(answer)

    # Sidebar buttons with improved layout and icons
    st.sidebar.markdown("### Actions")
    
    col3, col4 = st.sidebar.columns(2)

    with col3:
        align_button = st.button("✔️ Align", key="align_button")
    with col4:
        not_align_button = st.button("❌ Not Align", key="not_align_button")

    if align_button:
        item["user_choice"] = "align"
        save_data(data, save_file_path)
    if not_align_button:
        item["user_choice"] = "not_align"
        save_data(data, save_file_path)
    show_stats_button = st.sidebar.button("📊 Show Stats")
if show_stats_button:
    total_items = len(data)
    align_count = sum(1 for item in data if item.get("user_choice") == "align")
    user_choice_count = sum(1 for item in data if "user_choice" in item)
    align_rate = (align_count / user_choice_count) * 100 if user_choice_count > 0 else 0
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <p>Total items: {total_items}</p>
            <p>Items with user choices: {user_choice_count}</p>
            <p>Align rate: {align_rate:.2f}%</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
