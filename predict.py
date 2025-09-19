from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from geopy.geocoders import Nominatim
import shapefile
from shapely.geometry import Point, MultiLineString, shape
import re

def clean_tweet_text(t):
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@[a-zA-Z0-9_]+", "", t)
    t = re.sub(r"#", "", t)
    return str(t).strip() if t else ""

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

geolocator = Nominatim(user_agent="ocean_hazard_app_v2", timeout=10)
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

t_tweets = [
    "In the forest of fiat, #Bitcoin is the sacred fire. https://t.co/o1vKerHsAR",
    "Tsunami warning issued after 7.4 magnitude earthquake strikes off Alaska https://t.co/0gk3d8w1vD",
    "The moment a journalist broke down in tears while talking about the tragedy that claimed the lives of 1,000 Sudanese due to a landslide caused by floods in the village of Tarsin This country is unjustly forgotten by all https://t.co/qA2O3thJre",
    "Massive high waves hitting mumbai shores . 997e70d6-4d68-4e3e-bc12-a0d577ae5718",
    "Massive high waves hitting Alibaug shores . 997e70d6-4d68-4e3e-bc12-a0d577ae5718"
]

print("🧪 Making predictions on test tweets:")
print("-" * 50)

for i, t in enumerate(t_tweets, 1):
    print(f"\n{i}. Tweet: {t}")
    
    cleaned_t = clean_tweet_text(t)
    
    c_res = c(cleaned_t)
    ner_res = ner_p(t)
    
    print("   Classification:")
    highest_disaster = ""
    max_score = 0.0
    for l_s in c_res[0]:
        if l_s['score'] > 0.60:
            print(f"      - {l_s['label']}: {l_s['score']:.3f}")
            if l_s['score'] > max_score and l_s['label'] in oceanic_disasters:
                max_score = l_s['score']
                highest_disaster = l_s['label']

    if not highest_disaster:
        for d_k in oceanic_disasters:
            if d_k.replace("_", " ") in cleaned_t.lower():
                highest_disaster = d_k
                print(f"      - Keyword Fallback Detected: {highest_disaster}")
                break

    print("   Location:")
    locations = []
    for e in ner_res:
        if e['entity_group'] == 'LOC':
            loc_word = e['word'].strip()
            if loc_word:
                locations.append(loc_word)
                print(f"      - Detected Location: {loc_word} with score {e['score']:.3f}")
    
    if not locations:
        print("      - No location found.")
    
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
            print(f"      - Geocoding error for '{loc_name}': {e}")
            geo_data = None

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
                
            print(f"      - Location: {geo_data.address} ({dist_km:.1f} km from coast)")
            print(f"      - ☢️  Calculated Risk Level: {risk_level}")
        else:
            print(f"      - Could not verify location for risk assessment.")
    else:
        print("      - No actionable oceanic disaster detected to assess risk.")