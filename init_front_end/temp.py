import gensim.downloader
import time
print(list(gensim.downloader.info()['models'].keys()))
stopper = time.time()
model = gensim.downloader.load('glove-wiki-gigaword-300')
print(time.time()-stopper)
stopper= time.time()
print(model.most_similar('watch'))

print(time.time()-stopper)
