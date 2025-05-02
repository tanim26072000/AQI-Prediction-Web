import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# 1) PAGE CONFIG — MUST BE FIRST STREAMLIT CALL
st.set_page_config(page_title="AQI Predictor", layout="wide")

# 2) MODEL LOADING
MODEL_PATH = "model.h5"


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(
        optimizer='adam',
        loss='mean_absolute_error',
        metrics=['mean_squared_error', tf.keras.metrics.RootMeanSquaredError()]
    )
    return model


model = load_model()

# 3) PREPROCESSING


def preprocess_image(image: np.ndarray) -> tf.Tensor:
    img = tf.image.resize(image, (200, 200))
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)
    elif img.shape[-1] not in (1, 3):
        img = tf.expand_dims(img, axis=-1)
        img = tf.image.grayscale_to_rgb(img)
    img = img / 255.0
    return tf.ensure_shape(img[:120], (120, 200, 3))

# 4) GAUGE


def show_aqi_gauge(aqi_value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        domain={'x': [0, 1], 'y': [0, 0.8]},        # full vertical span
        number={
            'font': {'size': 28, 'color': 'white'}
        },
        gauge={
            'shape': 'angular',
            'axis': {
                'range': [0, 500],
                'tickmode': 'array',
                'tickvals': [25, 75, 125, 175, 225, 400],
                'ticktext': ['Good', 'Mod', 'USG', 'Unhealthy', 'VUnhealthy', 'Hazard'],
                'tickfont': {'size': 16, 'color': 'white'},
                'ticklen': 12,
                'tickwidth': 2,
                'tickcolor': 'white'
            },
            'steps': [
                {'range': [0, 50],    'color': '#00e400'},
                {'range': [50, 100],  'color': '#ffff00'},
                {'range': [100, 150], 'color': '#ff7e00'},
                {'range': [150, 200], 'color': '#ff0000'},
                {'range': [200, 300], 'color': '#8f3f97'},
                {'range': [300, 500], 'color': '#7e0023'},
            ],
            'bar': {'color': 'darkblue'},
        }
    ))

    fig.update_layout(
        height=300,                              # increase overall height
        margin={'t': 20, 'b': 0, 'l': 0, 'r': 0},  # give a bit of top margin
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )

    st.plotly_chart(fig, use_container_width=True)




# 5) LAYOUT
# header
left, center, right = st.columns([1, 2, 1])
with center:
    st.markdown("<h1 style='text-align:center;'>Air Quality Index (AQI) Predictor</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Upload a sky image to get its AQI.</p>",
                unsafe_allow_html=True)

# # legend
# st.markdown(
#     """
#     <div style="display:flex; justify-content:space-between; margin:20px 0;">
#       <div style="flex:1; text-align:center; background:#00e400; padding:5px;">0–50<br>Good</div>
#       <div style="flex:1; text-align:center; background:#ffff00; padding:5px;">51–100<br>Moderate</div>
#       <div style="flex:1; text-align:center; background:#ff7e00; padding:5px;">101–150<br>USG</div>
#       <div style="flex:1; text-align:center; background:#ff0000; padding:5px; color:#fff;">151–200<br>Unhealthy</div>
#       <div style="flex:1; text-align:center; background:#8f3f97; padding:5px; color:#fff;">201–300<br>V.Unhealthy</div>
#       <div style="flex:1; text-align:center; background:#7e0023; padding:5px; color:#fff;">301–500<br>Hazardous</div>
#     </div>
#     """, unsafe_allow_html=True
# )

# uploader + inference
uploaded = st.file_uploader("Choose an image…")
if uploaded:
    image = Image.open(uploaded)
    arr = np.array(image)
    prep = preprocess_image(arr)
    pred = float(model.predict(tf.expand_dims(prep, 0))[0][0])

    disp_w = 800
    # compute the display height so the aspect ratio is preserved
    disp_h = int(image.height * disp_w / image.width)

    col_img, col_txt = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Uploaded Image", width=disp_w)

    # decide color & status
    aqi = int(pred)
    if aqi <= 50:
        color, status = "#00e400", "Good"
    elif aqi <= 100:
        color, status = "#ffff00", "Moderate"
    elif aqi <= 150:
        color, status = "#ff7e00", "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        color, status = "#ff0000", "Unhealthy"
    elif aqi <= 300:
        color, status = "#8f3f97", "Very Unhealthy"
    else:
        color, status = "#7e0023", "Hazardous"

    with col_txt:
        # flex‐box container with exact height = disp_h
        st.markdown(f"""
<div style="
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: {disp_h}px;
">
  <div style="font-size:72px; font-weight:bold; color:{color};">
    {aqi}
  </div>
  <div style="font-size:24px; color:{color}; margin-top:5px;">
    {status}
  </div>
</div>
""", unsafe_allow_html=True)




    # ── Row 2: Gauge centered below ──
        # ── Row 2: Gauge and AQI Category Legend Side by Side ──
    c1, col_gauge, col_legend, c4 = st.columns([1, 2, 2, 1])
    with col_gauge:
        show_aqi_gauge(pred)

    with col_legend:
        st.markdown(
            """
            <div style="font-size:14px; line-height:1.5;">
              <strong>AQI Categories & Health Impacts</strong>
              <ul style="list-style:none; padding-left:0; margin-top:5px;">
                <li><span style="color:#00e400;">0–50 (Good)</span>: Air quality is satisfactory, and air pollution poses little or no risk.</li>
                <li><span style="color:#ffff00;">51–100 (Moderate)</span>: Acceptable; some pollutants may pose a moderate health concern for very sensitive people.</li>
                <li><span style="color:#ff7e00;">101–150 (Unhealthy for Sensitive Groups)</span>: Sensitive groups may experience health effects; general public unlikely to be affected.</li>
                <li><span style="color:#ff0000;">151–200 (Unhealthy)</span>: Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects.</li>
                <li><span style="color:#8f3f97;">201–300 (Very Unhealthy)</span>: Health alert—everyone may experience more serious health effects.</li>
                <li><span style="color:#7e0023;">301–500 (Hazardous)</span>: Health warnings of emergency conditions. The entire population is more likely to be affected.</li>
              </ul>
              <p style="font-size:12px; color:#888;">
                Source: <a href="https://aqicn.org/faq/2013-09-09/revised-pm25-aqi-breakpoints/" target="_blank">AQICN AQI Breakpoints</a>
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ── Optional LIME ──
    if st.checkbox("Show LIME explanation"):
        explainer = lime_image.LimeImageExplainer()
        exp = explainer.explain_instance(
            prep.numpy(),
            lambda x: model.predict(x),
            top_labels=1, hide_color=0, num_samples=1000
        )
        temp, mask = exp.get_image_and_mask(
            exp.top_labels[0], False, num_features=10, hide_rest=False)
        fig, ax = plt.subplots()
        ax.imshow(mark_boundaries(temp, mask))
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
