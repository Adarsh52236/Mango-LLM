import re

def fix_subwords(text):
    # A list of valid short English words to ignore when joining
    # We want to join subwords, so if a chunk is 1-2 letters and NOT a common word, we join it.
    
    # Actually, what if we use a regex that matches letters separated by space, 
    # where at least one of them is NOT a common 1-2 letter word?
    
    common_short = {"a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "hi", "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we"}
    
    # Split by spaces, but we want to retain punctuation spacing for now, or assume this runs after.
    words = text.split(" ")
    out = []
    
    for w in words:
        # if this word is a short non-common word, and the previous one was a letter ending, join?
        # A simpler rule: "Sp ie ge l" -> "Spiegel"
        # If we just do: `re.sub(r'([a-zA-Z]{1,2})\s+([a-zA-Z]{1,2})\b', r'\1\2', text)` repeatedly?
        pass

    # Let's try an iterative regex replacement:
    # We want to join two tokens if:
    # 1. They are both purely alphabetical.
    # 2. Together they don't form a situation where both were valid separate words.
    
    # Wait, BPE tokenizers usually produce subwords that are fragments. 
    # Maybe we can use a dictionary to check if the combined word is valid? 
    # No, BPE can produce names like "Spiegel".
    
    # What if we just join a 1-2 letter word with the previous/next if it's not a common short word?
    def replacer(match):
        w1, w2 = match.group(1), match.group(2)
        if w1.lower() in common_short and w2.lower() in common_short:
            return w1 + " " + w2
        # if one is a common word, maybe it shouldn't be joined?
        # e.g., "I am" -> "I am". "is it" -> "is it".
        # But what about "Sp ie"? "Sp" is not common, "ie" is not common. Join -> "Spie".
        # "Spie ge" -> "Spie" is 4 letters.
        
        # So we should actually join ANY letter sequence to an adjacent 1-2 letter sequence that is NOT a common word!
        return w1 + w2

    # Match a letter sequence followed by a space and a 1-2 letter sequence that is not common
    # or a 1-2 letter sequence not common followed by a space and a letter sequence.
    
    # Let's write a simple loop:
    tokens = re.split(r'(\s+)', text)
    
    # Actually, a simpler regex:
    # join single letters or non-word 2-letter chunks to adjacent words.
    
    # Let's see how this works:
    pass

text = "Sp ie ge l is a cat . I am a boy ."
# let's try the regex:
def clean(t):
    common_short_words = {'a', 'i', 'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we'}
    
    # repeatedly join
    changed = True
    while changed:
        changed = False
        # find word - space - word where at least one is a 1-2 letter non-word
        # we can use a regex findall
        matches = list(re.finditer(r'\b([a-zA-Z]+)\s+([a-zA-Z]+)\b', t))
        for m in matches:
            w1, w2 = m.group(1), m.group(2)
            # if one of them is <= 2 chars and not in common_short_words
            c1 = len(w1) <= 2 and w1.lower() not in common_short_words
            c2 = len(w2) <= 2 and w2.lower() not in common_short_words
            if c1 or c2:
                # join them!
                t = t[:m.start()] + w1 + w2 + t[m.end():]
                changed = True
                break # restart loop due to string length change
    return t

print("Original:", text)
print("Cleaned:", clean(text))
