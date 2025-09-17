import torch
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from geopy.geocoders import Nominatim
import shapefile
from shapely.geometry import Point, MultiLineString, shape
import re

app = Flask(__name__)

print("🚀 Initializing models and geospatial data...")

# --- Text Cleaning Function (copied from training script) ---
def clean_tweet_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[a-zA-Z0-9_]+", "", text)
    text = re.sub(r"#", "", text)
    return text.strip()

# --- Model Loading ---
m_path = "./trained_model"
tkn = AutoTokenizer.from_pretrained(m_path)
m = AutoModelForSequenceClassification.from_pretrained(m_path)

d = 0 if torch.cuda.is_available() else -1
c = pipeline("text-classification", 
                      model=m, 
                      tokenizer=tkn, 
                      device=d,
                      top_k=None)

ner_p = pipeline("ner", 
                        model="Jean-Baptiste/roberta-large-ner-english", 
                        device=-1, 
                        aggregation_strategy="simple")

# --- Geospatial Setup ---
geolocator = Nominatim(user_agent="ocean_hazard_app_v1")
sf = shapefile.Reader("ne_50m_coastline.shp")
lines = []
for s in sf.iterShapes():
    geom = shape(s)
    if geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            lines.append(line)
    else:
        lines.append(geom)
coastline = MultiLineString(lines)

location_cache = {}
oceanic_disasters = {"tsunami", "high_waves", "coastal_flooding"}
# --- End of Setup ---

print("✅ Server is ready to accept requests.")

@app.route("/analyze", methods=["POST"])
def analyze():
    j_data = request.get_json()
    if not j_data or 'text' not in j_data:
        return jsonify({"error": "Request must be JSON with a 'text' key."}), 400
    
    raw_text = j_data['text']
    # Apply the same cleaning as in training
    cleaned_text = clean_tweet_text(raw_text)

    # --- Run Models ---
    # Use cleaned text for the classifier, raw text for NER
    c_res = c(cleaned_text)
    ner_res = ner_p(raw_text)

    # --- Process Results ---
    classifications = []
    highest_disaster = ""
    max_score = 0.0
    for l_s in c_res[0]:
        if l_s['score'] > 0.65:
            classifications.append({"label": l_s['label'], "score": float(l_s['score'])})
            if l_s['score'] > max_score and l_s['label'] in oceanic_disasters:
                max_score = l_s['score']
                highest_disaster = l_s['label']

    locations = []
    for e in ner_res:
        if e['entity_group'] == 'LOC':
            locations.append({"location": e['word'], "score": float(e['score'])})

    # --- Geospatial Risk Assessment ---
    risk_assessment = {"status": "Not Applicable"}
    if highest_disaster and locations:
        loc_name = locations[0]['location']
        geo_data = None
        
        try:
            if loc_name in location_cache:
                geo_data = location_cache[loc_name]
            else:
                geo_data = geolocator.geocode(loc_name)
                location_cache[loc_name] = geo_data
        except Exception:
            pass

        if geo_data:
            loc_point = Point(geo_data.longitude, geo_data.latitude)
            dist_degrees = loc_point.distance(coastline)
            dist_km = dist_degrees * 111 

            risk_level = "LOW"
            if highest_disaster == "tsunami":
                if dist_km <= 20: risk_level = "CRITICAL"
                elif dist_km <= 50: risk_level = "HIGH"
            elif highest_disaster in ["high_waves", "coastal_flooding"]:
                if dist_km <= 30: risk_level = "HIGH"
            
            risk_assessment = {
                "status": "Assessed",
                "location_used": loc_name,
                "distance_km": round(dist_km, 2),
                "risk_level": risk_level
            }
        else:
            risk_assessment = {"status": "Location Not Verifiable"}

    # --- Final JSON Response ---
    final_response = {
        "inputText": raw_text,
        "classifications": classifications,
        "locations": locations,
        "riskAssessment": risk_assessment
    }
    
    return jsonify(final_response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)