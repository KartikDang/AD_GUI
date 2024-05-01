import pandas as pd
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer,util
import itertools
import ast
import operator
import streamlit as st
# import mymodel as m


desired_dimension = 768

df1 = pd.read_csv('keyword_headline_report_BROAD.csv')
df2 = pd.read_csv('keyword_headline_report_EXACT.csv')
df3 = pd.read_csv('keyword_headline_report_phrase.csv')

appended_df = pd.concat([df1,df2,df3])


df = appended_df

model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(model_name)


def findEmbeddings(df) :
  embeddings = []

  for adGroupName in df['ADGROUP_NAME']:
    if isinstance(adGroupName, str):
      currEmbedding = model.encode(adGroupName, convert_to_tensor = True)
      embeddings.append(currEmbedding.tolist())
    else:
      embeddings.append([])

  df['EMBEDDINGS'] = embeddings

  return df


df = findEmbeddings(df)

df.to_csv('embeddings.csv', index=False)



df = pd.read_csv('embeddings.csv')



def findEmbeddingOfParticularString(adGroupName):
    if isinstance(adGroupName, str):
        embedding = model.encode(adGroupName, convert_to_tensor=True).tolist()
        return embedding
    else:
        return []
    


def processEmbeddingsDataFrame (df, desired_dimension):
  embeddings = [np.pad(np.array((eval(embedding))), (0, desired_dimension - len((eval(embedding)))), 'constant')[:desired_dimension] for embedding in df['EMBEDDINGS']]
  return np.stack(embeddings)



def processEmbedding(embedding):
   return np.stack([np.pad(np.array(((embedding))), (0, desired_dimension - len(((embedding)))), 'constant')[:desired_dimension]])[0]


def similarityEmbedding (embedding1, embedding2):
  if torch.cuda.is_available():
    device = torch.device("cuda")
    # print("Using GPU for calculations.")
  else:
    device = torch.device("cpu")
    # print("Using CPU for calculations.")
  emb1 = torch.tensor(embedding1, dtype=torch.float32).to(device)
  emb2 = torch.tensor(embedding2, dtype=torch.float32).to(device)

  similarity = util.pytorch_cos_sim(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

  return similarity



def checkSimilarityForTwoDataFrames (df_test, df_train):
  # If Embeddings Already in File
  similarityScore = []
  # maximalMatching = []
  testEmbeddings = processEmbeddingsDataFrame(df_test, desired_dimension)
  trainEmbeddings = processEmbeddingsDataFrame(df_train, desired_dimension)

  for embedding in testEmbeddings:
    # maximalMatch
    maxSimi = -1
    maximalMatch = ""
    for trainEmbedding in trainEmbeddings:
      simi = similarityEmbedding(embedding, trainEmbedding)
      if(simi>maxSimi) :
        maxSimi = simi
        maximalMatch = trainEmbedding


    similarityScore.append((embedding, maximalMatch, maxSimi))


  result_df = pd.DataFrame(similarityScore,columns = ['Test Embedding','Train Embedding','CosineSimi']);
  return result_df


def checkSimilaritySpecificWithDataFrame(df, string):

  dfEmbeddings = processEmbeddingsDataFrame(df,desired_dimension)
  stringEmbedding = processEmbedding(findEmbeddingOfParticularString(string))

  # for embedding in dfEmbeddings:
  #   maxSimi = -1
  #   maximalMatch = ""
  #   for emb
  maximalMatch = ""
  maxSimi = -1
  for embedding in dfEmbeddings:
    simi = similarityEmbedding(stringEmbedding, embedding)
    if(simi>maxSimi):
      maxSimi = simi
      maximalMatch = embedding

  print(maxSimi)
  return maximalMatch, maxSimi



# Greater than 90% matching
def checkSimilaritySpecificWithDataFrame90Match(df, string):

  dfEmbeddings = processEmbeddingsDataFrame(df,desired_dimension)
  stringEmbedding = processEmbedding(findEmbeddingOfParticularString(string))

  # for embedding in dfEmbeddings:
  #   maxSimi = -1
  #   maximalMatch = ""
  #   for emb
  # maximalMatch = ""
  # maxSimi = -1
  embeddings = []
  for embedding in dfEmbeddings:
    simi = similarityEmbedding(stringEmbedding, embedding)
    # if(simi>maxSimi):
      # maxSimi = simi
      # maximalMatch = embedding
    if(simi>=0.9):
      embeddings.append((simi,embedding))

  # print(maxSimi)
  return embeddings



processEmbeddingsDataFrame(df,desired_dimension)


# Input Starts ->
def func (s):
  return (str(s)).replace('+','')

def processListOfInputKeywords(list):

    inputKeyword = [[keyword] for keyword in list]

    return inputKeyword


def processKeywordEmbeddingsDataFrame (df, desired_dimension):
  embeddings = [np.pad(np.array(((embedding))), (0, desired_dimension - len(((embedding)))), 'constant')[:desired_dimension] for embedding in df['EMBEDDINGS_KEYWORD']]
  return np.stack(embeddings)



def findCorrespondingName(embed, df, colNumber):

  for row in df.iterrows():
    embedding = row[1][colNumber]
    val = [np.pad(np.array(((embedding))), (0, desired_dimension - len(((embedding)))), 'constant')[:desired_dimension]]

    if((val == embed).all()):
      return row
    

def checkKeywordSimilarityForTwoDataFrames (dfInput, dfMain):
  matchedKeywords = {}  
  inputEmbeddings = processKeywordEmbeddingsDataFrame(dfInput, desired_dimension)
  mainEmbeddings = processKeywordEmbeddingsDataFrame(dfMain, desired_dimension)

  similarity_scores = []
  for embed in inputEmbeddings:
    cosineSimi = set()

    for mainEmbed in mainEmbeddings:
      simi = similarityEmbedding(embed, mainEmbed)
      keyword = findCorrespondingName(mainEmbed, dfMain, 45)
      # print(keyword)
      cosineSimi.add((keyword[1][7],keyword[1][15],keyword[1][20],keyword[1][22],keyword[1][23],keyword[1][24],keyword[1][29],simi))
      # break
    top_matches = sorted(cosineSimi, key=lambda x: x[1], reverse=True)[:10]
    # top_matches = cosineSimi
    keyword = findCorrespondingName(embed,dfInput,1)
    # matchedKeywords[embed] = top_matches
    # print(keyword[1][0])
    matchedKeywords[keyword[1][0]] = top_matches

def processResult(keywordInput, adGroupName):
    maxMatch = checkSimilaritySpecificWithDataFrame(df, adGroupName)

    # FINDING ADGROUP WITH SAME MATCHING EMBEDDING
    # labels
    for row in df.iterrows():
        embedding = row[1][44]
        val = [np.pad(np.array((eval(embedding))), (0, desired_dimension - len((eval(embedding)))), 'constant')[:desired_dimension]]

        if((val == maxMatch[0]).all()):
            label = row
            break

    mostSimilarADGroup = label[1][26]

    # Remove + from DF
    df['KEYWORD_TEXT'] = df['KEYWORD_TEXT'].apply(func)

    inputKeyword = processListOfInputKeywords(keywordInput)

    dfInput = pd.DataFrame(inputKeyword,columns = ["Keywords"])

    dfInput['EMBEDDINGS_KEYWORD'] = dfInput['Keywords'].apply(findEmbeddingOfParticularString)

    dfFiltered = df.loc[df['ADGROUP_NAME']==mostSimilarADGroup]

    dfFiltered['EMBEDDINGS_KEYWORD'] = dfFiltered['KEYWORD_TEXT'].apply(findEmbeddingOfParticularString)

    matchedKeywords = checkKeywordSimilarityForTwoDataFrames(dfInput, dfFiltered)

    overallScore = []
    scoreMap = {}
    for val in dfInput['Keywords']:
        # print(val)
        # print(matchedKeywords[val][0][0])

        currListOfList = matchedKeywords[val]
        tempList = []
        # print(currListOfList)
        SimilarityScore = 0
        for row in currListOfList:
            # print(row)
            salesByCost = row[3]/row[0]
            score = row[7]*salesByCost
            # print(salesByCost)
            SimilarityScore = SimilarityScore+score
            row=row+(salesByCost,score)


            # print(row)
            # break
            tempList.append(row)

            scoreMap[val] = SimilarityScore

        matchedKeywords[val] = tempList

        # break

    sorted_score = sorted(scoreMap.items(), key=operator.itemgetter(1), reverse=True)

        # Get the most profitable key
    most_profitable_key = sorted_score[0][0]

        # Print the most profitable key
    print(f"Most Profitable Key: {most_profitable_key}")

    return sorted_score


def takeInput(keywordInput, adGroupNameInput):

    keywordInput = keywordInput.split('\n')
    
    # Create a dictionary to hold the strings
    data = {"multiple_strings": keywordInput, "single_string": adGroupNameInput}
    
    # Convert the dictionary to JSON format
    res = processResult(keywordInput, adGroupNameInput)

    return res

def main():
    st.title("Input ADGroup Name and Kwywords")
    keywordsInput = st.text_area("Enter multiple keywords (one per line)")
    adGroupNameInput = st.text_input("Enter ADGroup Name")

    if keywordsInput and adGroupNameInput:
        with st.spinner("Generating JSON..."):
            # Generate JSON from inputs
            json_data = takeInput(keywordsInput, adGroupNameInput)
            
            # Display the generated JSON
            st.code(json_data, language='json')
    elif not keywordsInput and not adGroupNameInput:
        st.warning("Please enter some text in both input boxes.")

if __name__ == "__main__":
    main()