import csv
import random
from faker import Faker

def create_dataset(num_entries):
    h = ['text', 'location', 'tsunami', 'high_waves', 'coastal_flooding', 'not_relevant', 'panic', 'informational', 'help_needed']
    
    fake = Faker('en_IN')
    
    cl = ["Mumbai", "Chennai", "Vizag", "Puri", "Kochi", "Goa", "Alappuzha", "Gopalpur", "Marine Drive Mumbai", "Marina Beach Chennai", "Kovalam beach", "Digha", "Kollam", "Alibaug", "Nagoa Beach", "Cavelossim Beach"]
    ncl = ["New Delhi", "Bangalore", "Hyderabad", "Kolkata", "Pune", "Jaipur", "Ahmedabad", "Lucknow", "Chandigarh", "Mysuru", "Bhopal", "Agra", "Varanasi", "Amritsar"]

    e_p = ["😱", "😨", "😰", "😬", "🆘", "🏃‍♂️", "🚨"]
    e_i = ["⚠️", "📢", "📣", "📰", "🗓️", "✅", "✔️"]
    e_hn = ["🙏", "🚨", "🆘", "📞", "😭"]
    e_nr = ["😊", "😍", "😂", "😎", "👍", "🤷‍♀️", "🤦‍♂️", "😂"]

    def generate_unique_text(h_type, s_type, loc):
        if h_type == 'tsunami':
            base_phrases = [
                f"Tsunami warning issued for {loc}",
                f"Reports of a tsunami alert near {loc}",
                f"Evacuate coastal areas of {loc} immediately",
                f"The sea is receding at {loc}, a possible sign of tsunami",
                f"The news is confirming a tsunami threat for {loc}"
            ]
            template = random.choice(base_phrases)
        elif h_type == 'high_waves':
            base_phrases = [
                f"Massive high waves hitting {loc} shores",
                f"Rough sea conditions and strong waves in {loc}",
                f"Cyclone alert, waves getting dangerously high at {loc}",
                f"High tide and strong waves at {loc} beach",
                f"Unusually high waves are lashing the coasts of {loc}"
            ]
            template = random.choice(base_phrases)
        elif h_type == 'coastal_flooding':
            base_phrases = [
                f"Heavy waterlogging on roads in {loc}",
                f"Coastal flooding reported in {loc} after rains",
                f"Water is entering homes in {loc} near the shore",
                f"The streets of {loc} have turned into a river",
                f"Local authorities advising to avoid {loc} due to flooding"
            ]
            template = random.choice(base_phrases)
        elif h_type == 'not_relevant':
            base_phrases = [
                f"Enjoying the beautiful sunset at {loc}",
                f"My life is a tsunami of work",
                f"Had a great time at {loc}",
                f"The new movie's climax was a flood of emotions",
                f"Just chilling at {loc} beach",
                f"Great weather in {loc} today"
            ]
            template = random.choice(base_phrases)
        
        # Add a unique identifier to ensure text uniqueness
        unique_id = fake.uuid4()
        tweet_text = f"{template} {random.choice(['.', '!', '?'])} {unique_id}"
        return tweet_text

    def generate_entry(h_type, s_type):
        loc = random.choice(cl) if h_type != 'not_relevant' else random.choice(cl + ncl)
        tweet = generate_unique_text(h_type, s_type, loc)

        p, i, hn = 0, 0, 0
        if s_type == 'panic':
            p = 1
            tweet = f"{random.choice(e_p)} {tweet}"
        elif s_type == 'informational':
            i = 1
            tweet = f"{random.choice(e_i)} {tweet}"
        elif s_type == 'help_needed':
            p, hn = 1, 1
            tweet = f"{random.choice(e_hn)} {tweet}"
        elif s_type == 'neutral':
            tweet = f"{random.choice(e_nr)} {tweet}"

        t_label, hw_label, cf_label, nr_label = 0, 0, 0, 0
        if h_type == 'tsunami': t_label = 1
        elif h_type == 'high_waves': hw_label = 1
        elif h_type == 'coastal_flooding': cf_label = 1
        else: nr_label = 1
            
        return (tweet, loc, t_label, hw_label, cf_label, nr_label, p, i, hn)
    
    generated_data = []

    # Calculate exact counts for each category
    n_h = int(num_entries * 0.3)
    n_nr = num_entries - n_h
    n_t = int(n_h * 0.2)
    n_hw = int(n_h * 0.4)
    n_cf = n_h - n_t - n_hw
    
    # Generate balanced hazard tweets with correct sentiment distribution
    def generate_balanced_category(h_type, total_count, data_list):
        s_counts = {'panic': int(total_count * 0.4), 'informational': int(total_count * 0.4), 'help_needed': int(total_count * 0.2)}
        
        for s_type, count in s_counts.items():
            for _ in range(count):
                entry = generate_entry(h_type, s_type)
                data_list.append(entry)
    
    generate_balanced_category('tsunami', n_t, generated_data)
    generate_balanced_category('high_waves', n_hw, generated_data)
    generate_balanced_category('coastal_flooding', n_cf, generated_data)

    # Generate non-relevant tweets
    n_nr_p = int(n_nr * 0.05)
    n_nr_i = int(n_nr * 0.05)
    n_nr_n = n_nr - n_nr_p - n_nr_i
    
    for _ in range(n_nr_p): generated_data.append(generate_entry('not_relevant', 'panic'))
    for _ in range(n_nr_i): generated_data.append(generate_entry('not_relevant', 'informational'))
    for _ in range(n_nr_n): generated_data.append(generate_entry('not_relevant', 'neutral'))
    
    random.shuffle(generated_data)
    
    with open('ocean_hazards_tweets.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(h)
        w.writerows(generated_data)

    print(f"Generated {len(generated_data)} unique and balanced lines of synthetic tweet data into 'ocean_hazards_tweets.csv'")

if __name__ == "__main__":
    create_dataset(20000)