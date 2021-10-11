# !pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
# !pip install sentencepiece

#1. Import and Load Model
# Importing dependencies from transformers
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Tokenizer converts sentences into tokens (# representation for words)
# PegasusForConditionalGeneration = allows for model setup

# Load tokenizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

# Load model
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

text = """

Papua New Guinea (PNG; /ˈpæp(j)uə ... ˈɡɪni, ˈpɑː-/, also US: /ˈpɑːpwə-, ˈpɑːpjə-, ˈpɑːpə-/;[12] Tok Pisin: Papua Niugini; Hiri Motu: Papua Niu Gini; Torres Strait Creole: Dhaudhai Niu Gini; Meriam Mir: Op Deudai),[13] officially the Independent State of Papua New Guinea (Tok Pisin: Independen Stet bilong Papua Niugini; Hiri Motu: Independen Stet bilong Papua Niu Gini), is a country in Oceania that comprises the eastern half of the island of New Guinea and its offshore islands in Melanesia (a region of the southwestern Pacific Ocean north of Australia). Its capital, located along its southeastern coast, is Port Moresby. It is the world's third largest island country with an area of 462,840 km2 (178,700 sq mi).[14]

At the national level, after being ruled by three external powers since 1884, Papua New Guinea established its sovereignty in 1975. This followed nearly 60 years of Australian administration, which started during World War I. It became an independent Commonwealth realm in 1975 with Elizabeth II as its queen. It also became a member of the Commonwealth of Nations in its own right.

Papua New Guinea is one of the most culturally diverse countries in the world. As of 2019, it is also the most rural, as only 13.25% of its people live in urban centres.[15] There are 851 known languages in the country, of which 11 now have no known speakers.[6] Most of the population of more than 8,000,000 people live in customary communities, which are as diverse as the languages.[16] The country is one of the world's least explored, culturally and geographically.[by whom?][citation needed] It is known to have numerous groups of uncontacted peoples, and researchers believe there are many undiscovered species of plants and animals in the interior.[17]

The sovereign state is classified as a developing economy by the International Monetary Fund.[18] Nearly 40% of the population lives a self-sustainable natural lifestyle with no access to global capital.[19] Most of the people live in strong traditional social groups based on farming. Their social lives combine traditional religion with modern practices, including primary education.[16] These societies and clans are explicitly acknowledged by the Papua New Guinea Constitution, which expresses the wish for "traditional villages and communities to remain as viable units of Papua New Guinean society"[20] and protects their continuing importance to local and national community life. The nation is an observer state in the Association of Southeast Asian Nations (ASEAN) since 1976 and has filed its application for full membership status.[21] It is a full member of the Pacific Community, the Pacific Islands Forum,[22] and the Commonwealth of Nations.[23]

"""

# Create tokens - number representation of our text
tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
# truncation to shorten text
# padding
# return_tensors = pytorch tenses

# Input tokens
tokens

# Summarize
summary = model.generate(**tokens) # **unpacks the tokens

# This is a output summary tokens
summary[0]

# Decode summary; perform a decoding step
png = tokenizer.decode(summary[0])
print(png)
