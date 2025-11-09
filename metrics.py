import re, math, pandas as pd, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
import nltk, sys, os
nltk.download('punkt')

def load_dialog(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

def entropy(labels):
    freq = pd.Series(labels).value_counts(normalize=True)
    return -freq.apply(lambda x: x*math.log2(x)).sum()

def odi(texts):
    """Topic-diversity entropy across speakers"""
    vec = CountVectorizer(stop_words='english', max_features=500)
    mx = vec.fit_transform(texts)
    topic_dist = mx.sum(axis=0).A1
    topic_dist = topic_dist / topic_dist.sum()
    return -sum(p*math.log2(p) for p in topic_dist if p)

def adi(texts):
    """Argument diversity = balance pro/con sentences"""
    # قاعدة بسيطة (يمكن تحسينها لاحقاً)
    pro_words  = {'can', 'will', 'yes', 'innovative', 'creative', 'possible'}
    con_words  = {'cannot', 'will not', 'no', 'mere', 'simulation', 'not real'}
    scores = []
    for txt in texts:
        tokens = re.findall(r'\b\w+\b', txt.lower())
        pro  = sum(1 for w in tokens if w in pro_words)
        con  = sum(1 for w in tokens if w in con_words)
        total = pro + con or 1
        balance = 1 - abs(pro - con) / total   # 1 = perfect balance
        scores.append(balance)
    return sum(scores)/len(scores)

def main(md_file):
    raw = load_dialog(md_file)
    # افتراض تقسيم بسيط حسب المشارك
    chunks = re.split(r'\n## ', raw)
    human, generative, critical = [], [], []
    for chk in chunks:
        if chk.startswith('Human'):
            human.append(chk)
        elif chk.startswith('Generative'):
            generative.append(chk)
        elif chk.startswith('Critical'):
            critical.append(chk)
    texts = [' '.join(human), ' '.join(generative), ' '.join(critical)]
    odi_score = odi(texts)
    adi_score = adi(texts)
    print(f"ODI (Topic Diversity) : {odi_score:.3f}")
    print(f"ADI (Argument Balance): {adi_score:.3f}")
    # رسم سريع
    plt.bar(['ODI','ADI'], [odi_score, adi_score])
    plt.ylim(0,1); plt.title("Al-Jahiz PoC Metrics")
    plt.savefig("chart.png"); plt.close()
    print("chart.png saved.")

if __name__ == '__main__':
    file = sys.argv[1] if len(sys.argv)>1 else 'sample.md'
    main(file)
