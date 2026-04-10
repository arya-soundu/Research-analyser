import re
#keyword dictionary
KEYWORDS = {
    "objective": [
        "aim", "goal", "objective", "this paper", "this study",
        "we propose", "we present", "investigate", "explore", "purpose",
        "our approach", "our method", "we introduce", "motivation"
    ],
    "methodology": [
        "method", "approach", "dataset", "experiment", "we use",
        "we train", "architecture", "using", "based on", "algorithm",
        "clustering", "embedding", "tokenizer", "framework", "pipeline",
        "we collected", "training", "fine-tuned"
    ],
    "findings": [
        "result", "found", "show", "demonstrate", "achieve",
        "accuracy", "performance", "outperform", "rouge", "score",
        "our model", "compared to", "highest", "lowest", "table",
        "figure", "evaluation", "metric"
    ],
    "conclusion": [
        "conclude", "conclusion", "future work", "in summary",
        "limitation", "contribute", "overall", "we believe",
        "further", "scope", "enhance", "improve", "future"
    ]
}

#Keyword heuristics -> look for signal words based on domain knowledge 
#Scoring each section based on keyword matches from KEYWORDS
def classify_sentence(sentence:str)->(str):
    #lowercase the sentence for easier matching
    lower=sentence.lower()
    scores = {section: 0 for section in KEYWORDS}
    for section,words in KEYWORDS.items():
        for w in words:
            if w in lower:
                scores[section]+=1
    # strong signal override — if these phrases appear, classify immediately
    # without even checking scores. These are unambiguous section markers.
    strong_signals = {
    "conclusion": ["in conclusion", "in summary", "to conclude", "future work", "we conclude"],
    "objective":  ["this paper proposes", "this study investigates the application"],  # ← more specific
    }

    for section, phrases in strong_signals.items():
        for phrase in phrases:
            if phrase in lower:
                return section   # return immediately, no need to check scores
    #return the section with the highest score
    return max(scores,key=scores.get) if max(scores.values()) > 0 else "findings"

#Split the summary into induvidual senetnces and then sent it to classify_sentence()
#Then build a dictionary with the sections as keys and the sentences as values 
def build_structured_notes(summary:str)->dict:
    sentences = re.split(r'(?<=[.!?]) +',summary.strip())
    sections = {"objective":[],
    "methodology":[],
    "findings":[],
    "conclusion":[]}
    for sentence in sentences:
        if sentence.strip():
            section=classify_sentence(sentence)
            sections[section].append(sentence)
    return {
        k: " ".join(v) if v else f"{k.capitalize()} not clearly identified."
        for k, v in sections.items()
    }