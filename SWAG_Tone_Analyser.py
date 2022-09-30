import streamlit as st
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 \
    import Features, CategoriesOptions, EmotionOptions, KeywordsOptions, EntitiesOptions

authenticator = IAMAuthenticator(st.secrets["API"])
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator
)

natural_language_understanding.set_service_url(st.secrets["URL"])


def getEmotions(data):
    emotions = []
    emotions_dict = {}
    emotions_scores = []
    response_emo = natural_language_understanding.analyze(
                    text = data,
                    features = Features( emotion=EmotionOptions())
                    ).get_result() 

    for i in list(response_emo['emotion']['document']['emotion'].keys()):
        emotions.append([i, str(response_emo['emotion']['document']['emotion'][i])])
        emotions_scores.append(str(response_emo['emotion']['document']['emotion'][i]))

    for i in list(response_emo['emotion']['document']['emotion'].keys()):
        emotions_dict[i] = str(response_emo['emotion']['document']['emotion'][i])   

    emotions_scores.sort()

    for e in emotions:
        if emotions_scores[-1] == e[1]:
            max_emo = e[0]    

    return emotions, max_emo, emotions_dict    

def getKeywords(data):
    Keywords = []
    Sentiments = []
    score = 0
    response_keyw = natural_language_understanding.analyze(
                    text = data,
                    features=Features(keywords=KeywordsOptions(sentiment=True, limit=10))).get_result()

    i = 0
    while i < len(response_keyw['keywords']):
        Keywords.append(response_keyw['keywords'][i]['text'])
        Sentiments.append(response_keyw['keywords'][i]['sentiment']['label'])
        score += response_keyw['keywords'][i]['sentiment']['score']
        i+=1
        
    return Keywords, Sentiments, score/(len(Keywords))


st.header("S.W.A.G Tone Analyser")

data = st.text_area('Analyse Your Text')

if st.button('Analyse'):
    emotionsList, max_emotion, emo_dict = getEmotions(data)
    st.success('Your text is analysed!')
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    for i in range(len(emotionsList)):
        a = emotionsList[i][0]
        emotionsList[i][0] = emotionsList[i][1]
        emotionsList[i][1] = a        
    
    emotionsList.sort(reverse=True)

    for i in emotionsList:
        col1.write(i[1])
        col2.write(i[0])

    keywords, sentiments, score = getKeywords(data)
    for i in keywords:
        col3.write(i)
    for j in sentiments:    
        col4.write(j)
    col5.write('Sentiment Score')
    col5.write(score)


    if score < -0.7:
        if max_emotion == 'joy' or emo_dict['joy'] > '0.55':
            col6.write('NOT HATEFUL')    
        elif max_emotion == 'anger' and emo_dict['fear'] > '0.05' and emo_dict['disgust'] > '0.05':
            col6.write('HATEFUL & Inflammatory')
        elif max_emotion == 'sadness' and emo_dict['anger'] > '0.1':
            col6.write('HATEFUL')
        elif max_emotion == 'disgust':
            if emo_dict['disgust'] > '0.2' and emo_dict['anger'] > '0.05':
                col6.write('HATEFUL')
            elif emo_dict['anger'] > '0.1':
                col6.write('HATEFUL') 
    elif emo_dict['disgust'] > '0.17' and emo_dict['anger'] > '0.05':
        col6.write('HATEFUL')
    elif max_emotion == 'sadness' and emo_dict['anger'] > '0.1':
            col6.write('HATEFUL')
    else:
        col6.write('NOT HATEFUL')


