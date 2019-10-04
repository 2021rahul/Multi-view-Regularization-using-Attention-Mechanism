import os
import numpy as np
import re
import tensorflow as tf


DIR = os.path.join("..", "DATA", "imdb")
wordList = np.load(os.path.join(DIR, "wordList.npy"))
wordVectors = np.load(os.path.join(DIR, "wordVectors.npy"))
wordList = wordList.tolist()
print(wordList.index(''))
print(wordVectors[wordList.index('unk')])

# with open(os.path.join(DIR, "glove.6B.100d.txt"), "r") as f:
#     words = f.readlines()
#
# wordList = []
# wordVectors = []
# for word in words:
#     elements = word.split()
#     wordList.append(elements[0])
#     vec = np.zeros((100))
#     for i in range(100):
#         vec[i] = float(elements[i+1])
#     wordVectors.append(vec)
#
# wordList = np.reshape(np.array(wordList),(-1,))
# wordVectors = np.reshape(np.array(wordVectors),(-1,100))
#
# np.save(os.path.join(DIR, "wordsList"), wordList)
# np.save(os.path.join(DIR, "wordVectors"), wordVectors)

# wordsList = np.load(os.path.join(DIR,'wordsList.npy'))
# print('Loaded the word list!')
# wordsList = wordsList.tolist() #Originally loaded as numpy array
# wordVectors = np.load(os.path.join(DIR,'wordVectors.npy'))
# print ('Loaded the word vectors!')
#
# print(len(wordsList))
# print(wordVectors.shape)
#
# baseballIndex = wordsList.index('baseball')
# wordVectors[baseballIndex]
#
# maxSeqLength = 10
# numDimensions = 300
# firstSentence = np.zeros((maxSeqLength), dtype='int32')
# firstSentence[0] = wordsList.index("i")
# firstSentence[1] = wordsList.index("thought")
# firstSentence[2] = wordsList.index("the")
# firstSentence[3] = wordsList.index("movie")
# firstSentence[4] = wordsList.index("was")
# firstSentence[5] = wordsList.index("incredible")
# firstSentence[6] = wordsList.index("and")
# firstSentence[7] = wordsList.index("inspiring")
# print(firstSentence.shape)
# print(firstSentence)
#
# with tf.Session() as sess:
#     print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)

# positiveFiles = os.listdir(os.path.join(DIR, "train", "pos"))
# positiveFiles = [file for file in positiveFiles if file.endswith(".txt")]
# negativeFiles = os.listdir(os.path.join(DIR, "train", "neg"))
# negativeFiles = [file for file in negativeFiles if file.endswith(".txt")]

# numWords = []
# numLines = []
# for file in positiveFiles:
#     with open(os.path.join(DIR, "train", "pos", file), "r", encoding='utf-8') as f:
#         line=f.readline()
#         counter = len(line.split())
#         numLines.append(len(line.split(".")))
#         numWords.append(counter)
# print('Positive files finished')
#
# for file in negativeFiles:
#     with open(os.path.join(DIR, "train", "neg", file), "r", encoding='utf-8') as f:
#         line=f.readline()
#         counter = len(line.split())
#         numLines.append(len(line.split(".")))
#         numWords.append(counter)
# print('Negative files finished')
#
# numFiles = len(numWords)
# print('The total number of files is', numFiles)
# print('The total number of words in the files is', sum(numWords))
# print('The average number of words in the files is', sum(numWords)/len(numWords))
# print('The total number of lines in the files is', sum(numLines))
# print('The average number of lines in the files is', sum(numLines)/len(numLines))

# fname = positiveFiles[3]
# with open(os.path.join(DIR, "train", "pos", fname), "r") as f:
#     for lines in f:
#         print(lines, "\n")
#         strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
#         lines = lines.lower().replace("<br />", " ")
