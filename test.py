import fasttext

en_model = fasttext.load_model('wiki.en/wiki.en.bin')

#Works 
embeddings = np.array([])
labels = []
i = 0
for word in en_model.words:
    i += 1
    labels.append(word)
    embeddings = np.append(embeddings, en_model[word])
    if i == 5000: break
    

# Works
embeddings = embeddings.reshape(5000, 300)
