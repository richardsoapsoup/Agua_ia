import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
from PIL import Image

st.set_page_config(page_title="Water Classification", page_icon="💧", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def extract_features(image):
    # Convert PIL image to OpenCV format (numpy array)
    img_array = np.array(image)
    
    # Check if image is RGB, if not, convert to RGB
    if len(img_array.shape) == 2:
        # Grayscale to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        # RGBA to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Convert from RGB to BGR to match the training data format
    # (The original dataset was likely generated with cv2.imread which loads as BGR, 
    # but the columns were named r, g, b)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    b, g, r = cv2.split(img_bgr)
    
    # Map B->r, G->g, R->b to match the CSV's swapped columns
    hist_r = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
    
    # Combine into a single feature vector matching the CSV format
    features = np.concatenate((hist_r, hist_g, hist_b))
    
    # Normalize features (L1 norm) to be independent of image size
    total_pixels = np.sum(features)
    if total_pixels > 0:
        features = features / total_pixels
        
    return features

def main():
    st.title("💧 Classificador de Água: Limpa ou Suja?")
    st.write("Faça o upload da foto de um copo de água e descubra se a água está limpa ou suja, baseado em análise de cor (Histograma RGB).")
    
    model = load_model()
    
    if model is None:
        st.warning("O modelo ainda não foi treinado. Por favor, aguarde o treinamento ou verifique os arquivos.")
        return
        
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagem original.', width='stretch')
        
        use_crop = st.checkbox("Aplicar recorte central (focar no copo e ignorar fundo)", value=True)
        
        if use_crop:
            # Recorta os 50% centrais da imagem usando Pillow
            w, h = image.size
            start_w, end_w = int(w * 0.25), int(w * 0.75)
            start_h, end_h = int(h * 0.25), int(h * 0.75)
            image = image.crop((start_w, start_h, end_w, end_h))
            st.image(image, caption='Imagem recortada (usada para extração das cores).', width='stretch')
        
        st.write("---")
        st.write("### 🎛️ Calibração de Ambiente")
        st.write("O modelo foi treinado com imagens em condições de luz específicas. Se a sua foto estiver muito mais clara ou mais escura que as do treinamento, você pode ajustar o brilho para calibrar a predição.")
        
        brightness = st.slider("Ajuste de Brilho (tente valores negativos para fundos muito claros)", -100, 100, 0)
        
        if brightness != 0:
            # Converte a imagem PIL para OpenCV para aplicar o brilho
            img_cv = np.array(image)
            img_cv = cv2.convertScaleAbs(img_cv, alpha=1, beta=brightness)
            image = Image.fromarray(img_cv)
            st.image(image, caption=f'Imagem com ajuste de brilho ({brightness}).', width='stretch')

        st.write("")
        st.write("Extraindo características e classificando...")
        
        # Extract features
        features = extract_features(image)
        
        # We need to reshape for a single prediction and create a DataFrame to avoid warnings about feature names if sklearn expects them
        # Let's create the column names matching the original CSV
        columns = [f'r{i}' for i in range(256)] + [f'g{i}' for i in range(256)] + [f'b{i}' for i in range(256)]
        features_df = pd.DataFrame([features], columns=columns)
        
        # Predict
        prediction = model.predict(features_df)
        
        # Display result
        result_class = prediction[0]
        if result_class.lower() == 'limpo':
            st.success(f"O modelo classificou a água como: **{result_class.upper()}** ✅")
        else:
            st.error(f"O modelo classificou a água como: **{result_class.upper()}** ❌")
            
        st.write("---")
        with st.expander("Ver Histograma RGB"):
            st.line_chart(pd.DataFrame({
                'Blue': features[:256],
                'Green': features[256:512],
                'Red': features[512:]
            }))

if __name__ == '__main__':
    main()
