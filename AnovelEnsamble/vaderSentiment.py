# VaderSentiment is an opensource library on MIT lic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# function to print sentiments
# of the sentence.
def vader_sentiment(sentence):

	sid_obj = SentimentIntensityAnalyzer()

	sentiment_dict = sid_obj.polarity_scores(sentence)

	print("Overall sentiment dictionary is : ", sentiment_dict)
	print(sentiment_dict['neg']*100, "% Negative")
	print(sentiment_dict['neu']*100, "% Neutral")
	print(sentiment_dict['pos']*100, "% Positive")

	print("Overall", end = " ")

	# decide sentiment as positive, negative and neutral
	if sentiment_dict['compound'] >= 0.05 :
		print("Positive")

	elif sentiment_dict['compound'] <= - 0.05 :
		print("Negative")

	else :
		print("Neutral")

  


# Driver code
if __name__ == "__main__" :

	print("\n1st statement :")
	sentence = "You are the worst"
	# function calling
	a=vader_sentiment(sentence)
# -*- coding: utf-8 -*-

