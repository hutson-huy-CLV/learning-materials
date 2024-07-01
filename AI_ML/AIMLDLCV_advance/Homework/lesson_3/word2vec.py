import gensim.downloader as api
import numpy as np

# 25, 50, 100 or 200. Số càng lớn thì càng chính xác, nhưng chạy càng lâu các bạn nhé
model = api.load("glove-twitter-200")
word = "scary"
print(model[word])

print("1----------")
result = model.most_similar(word, topn=10)
print(result)

print("2----------")
result = model.most_similar(positive=["first", "second"], topn=10)
print(result)

print("3----------")
result = model.similarity("man", "boy")
print(result)

print("4----------")
result = model.most_similar(positive=["woman", "king"], negative=["man"], topn=1)
print(result)

print("5----------")
result = model.most_similar(positive=["berlin", "vietnam"], negative=["hanoi"], topn=1)
print(result)

print("6----------")
result = model.similarity("marriage", "happiness")
print(result)

print("7----------")
result = model.similarity("marriage", "unhappiness")
print(result)



# TODO: Các bạn thử viết 2 cách khác nhau để tính cosine similarity
# giữa 2 vector nhé. Kết quả giống với khi dùng model.similarity() là được
# 1 cách là dùng numpy, 1 cách là không dùng numpy nhé
