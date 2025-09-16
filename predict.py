# predict.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from geopy.geocoders import Nominatim
import shapefile
from shapely.geometry import Point, MultiLineString, shape

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

t_tweets = [
    "Tsunami warning issued for Chennai. Evacuate immediately!",
    "Massive waves hitting Mumbai beach right now! Very scary situation.",
    "Hearing reports about a potential tsunami in Pune, seems fake.",
    "Water entering homes in coastal areas of Puri. Need help urgently."
]

print("🧪 Making predictions on test tweets:")
print("-" * 50)

for i, t in enumerate(t_tweets, 1):
    print(f"\n{i}. Tweet: {t}")
    
    c_res = c(t)
    ner_res = ner_p(t)
    
    print("   Classification:")
    highest_disaster = ""
    max_score = 0.0
    for l_s in c_res[0]:
        if l_s['score'] > 0.65:
            print(f"      - {l_s['label']}: {l_s['score']:.3f}")
            if l_s['score'] > max_score and l_s['label'] in oceanic_disasters:
                max_score = l_s['score']
                highest_disaster = l_s['label']

    # --- Location logging restored ---
    print("   Location:")
    locations = []
    for e in ner_res:
        if e['entity_group'] == 'LOC':
            locations.append(e['word'].strip())
            print(f"      - Detected Location: {e['word']} with score {e['score']:.3f}")
    
    if not locations:
        print("      - No location found.")
    
    # --- Geospatial Risk Assessment (Detailed Output) ---
    print("   Risk Assessment:")
    if highest_disaster and locations:
        loc_name = locations[0]
        geo_data = None
        
        try:
            if loc_name in location_cache:
                geo_data = location_cache[loc_name]
            else:
                geo_data = geolocator.geocode(loc_name)
                location_cache[loc_name] = geo_data
        except Exception as e:
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

            # --- Distance logging restored ---
            print(f"      - Location: {loc_name} ({dist_km:.1f} km from coast)")
            print(f"      - ☢️  Calculated Risk Level: {risk_level}")
        else:
            print(f"      - Could not verify location.")
    else:
        print("      - No oceanic disaster detected to assess risk.")