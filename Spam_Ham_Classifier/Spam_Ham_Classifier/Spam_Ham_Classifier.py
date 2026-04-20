import pandas as pd #for reading .csv files
import matplotlib.pyplot as plt #for barchart
import heapq  #for taking greatest value from dictionary, to make lists with top words for barchart
import re #for turning links & telephon numbers into respectively linkToken & phoneNumberToken tokens
from nltk.stem import PorterStemmer #for stemming words
import math





#This classifier uses classical Bernoulli Naive Bayes with Laplace Smoothing

#First I redacted original .csv. Just changing column names for more appropriate ones and turning spam/ham into boolean variables:

'''
SMS = (pd.read_csv("spam_ham.csv", encoding = "latin-1"))[["v1", "v2"]]             #reading it
SMS = SMS.rename(columns = {'v1': 'status', 'v2': 'text'})                          #renaming columns
SMS["status"] = SMS["status"].map({"ham" : True, "spam" : False})                   #turns ham/spam to boolean values
SMS = SMS[SMS['text'].str.strip().astype(bool)]                                     #removes empty rows
SMS.to_csv("spamCleaned.csv", index=False, encoding="latin-1")                      #save changes in a new .csv
'''



SMS = pd.read_csv("spamCleaned.csv", encoding = "latin-1", nrows = 5000) #dataframe with first 5000 rows(training set), rest is going to be used for testing


#Now analyzing the data, for this I used charts to see which words appear most frequently
#After analyzing I came to conclusion that to make data usable I need to do the following:

#1. make all words lowercase
#2. get rid of punctuation
#3. remove words that are single characters
#4. words stemming, turn words with the same root into a single word, for instance: running, runner, runs => run
#5. remove all the words that don't tell anything like 'the', 'such', 'this', 'any'
#6. get rid of all the unpopular words

#Here's the process of doing so + creating bar charts


#numberOfEachWord = {} #for words from all messages with word => key and their frequency => value
#spamWordsFrq = {}     #for words from spam messages
#hamWordsFrq = {}      #for words from ham messages


#stemmer = PorterStemmer() #used for stemming words
#for index, row in SMS.iterrows(): #SMS.iterrows() gives two values: index and row, which row could be specified later
#    if index % 100 == 0:  #checks if it's still running
#        print("It's ", index, "'s iteration")

#    message = row["text"].lower() #takes message from "text" column in lowercase
#    message = re.sub(r'http[s]?://\S+', 'linkToken', message) # turns links into linkTokens
#    message = re.sub(r'\+?\d[\d\s\-\(\)]{5,}\d', 'phoneNumberToken', message) # turns phone numbers into a phoneNumberToken \+? checks for + in the beginning, \d checks if starts with number, [\d\s\-\(\)], checks if continituous with \digit(number)\space\-\(\) so it would be good for 12345678 123-456-7890 (123) 456 7890 +49 157 1234567 and so on, {7,} checks if it's at least 7 digits long, \d checks if ends in digit
#    message = message.split() #turns string into a list that consists out of each words in this very message
#    message = [stemmer.stem(word) for word in message] # stemming it
#    message_stripped = [word for word in [word.strip(",.;:-_/'?!") for word in message] if word != '' and (len(word) > 1 and word != "2")] #removes punctuation and words that are one character long

#    if(row["status"] == False): #checks if this SMS is spam, False = spam, True = ham
#        for i in message_stripped:#goes trough every word
#            if i in spamWordsFrq:#checks if this word is already a key in dictionary
#                spamWordsFrq[i] = spamWordsFrq[i] + 1 #if so increment the value of this word(we've seen it once more)
#            else:
#                spamWordsFrq[i] = 1 #if no then create key with this word that corresponds to one(we saw this word for the first time)
#    else:
#        for i in message_stripped: #does the same for ham
#            if i in hamWordsFrq:
#                hamWordsFrq[i] = hamWordsFrq[i] + 1
#            else:
#                hamWordsFrq[i] = 1
                

#    for i in message_stripped: #does the same but for all messages combined
#        if i in numberOfEachWord:
#            numberOfEachWord[i] = numberOfEachWord[i]+1
#        else:
#            numberOfEachWord[i] = 1

##removing words that don't tell us anything
#for key in ["to", "you", "the", "and", "is", "in", "for", "your", "have", "it", "on", "of", "are", "or", "with", "get", "just"]:
#    del numberOfEachWord[key]
#    del spamWordsFrq[key]
#    del hamWordsFrq[key]

##removing very unpopular words to shrink dictionary size (now they only continue words that are used more than 2 times)
#numberOfEachWord = {key: value for key, value in numberOfEachWord.items() if value > 2} #all words dic
#spamWordsFrq = {key: value for key, value in spamWordsFrq.items() if value > 2}         #spam dic
#hamWordsFrq = {key: value for key, value in hamWordsFrq.items() if value > 2}           #ham dic

##taking top 40 most used words in both spam and ham for barcharts, returns list with tuples.
#top_values = heapq.nlargest(40, numberOfEachWord.items(), key=lambda x: x[1]) # heapq.nlargest returns a list of top 40 tuples sorted by largest number
#                                                                              # .items() returns list with tuples like so: {"hi": 121, "call": 1555} => [("hi", 121), ("call", 1555)]
#                                                                              # key=lambda x: x[1] tells it to sort them by second element in the tuple(so by largest numbers and not lexigraphically)
#top_spamValues = heapq.nlargest(40, spamWordsFrq.items(), key=lambda x: x[1])
#top_hamValues = heapq.nlargest(40, hamWordsFrq.items(), key=lambda x: x[1]) 

##same can be done for smallest values with heapq.nsmalles() but I don't think it will serve any purpose.

##creating lists that will contain top 40 words and values from top_values(words from all messages) to be used for bar charts.
#top_valuesFrequency = []
#top_valuesWords = []

#topSpamFrq = []  #same for words from spam messages
#topSpamWords = []

#topHamFrq = []  #same for ham messages
#topHamWords = []

#for word, value in top_spamValues: #assigning values to spam lists
#    topSpamFrq.append(value)
#    topSpamWords.append(word)

#for word, value in top_hamValues: #assigning values to ham lists
#    topHamFrq.append(value)
#    topHamWords.append(word)
    
#for word, value in top_values: #assignign values to all messages lists
#    top_valuesFrequency.append(value)
#    top_valuesWords.append(word)



#plt.figure() #creates new window
#plt.barh(topSpamWords, topSpamFrq, color='red') #creates bar chart with first argument being labels, second values

#plt.figure()
#plt.barh(topHamWords, topHamFrq, color='blue')

#plt.figure()
#plt.barh(top_valuesWords, top_valuesFrequency, color='yellow')

#plt.show() #shows everything that was done in plt.


#Now that I've found the optimal way to use data, I can just use it combined with classial Bernoulli Naive Bayes and have my classifier

numberOfEachWord = {} #for words from all messages with word => key and their frequency => value
spamWordsFrq = {}     #for words from spam messages
hamWordsFrq = {}      #for words from ham messages

stemmer = PorterStemmer() #used for stemming words
for index, row in SMS.iterrows(): #SMS.iterrows() gives two values: index and row, which row could be specified later
    if index % 100 == 0:  #checks if it's still running#
        print("It's ", index, "'s iteration")

    message = row["text"].lower() #takes message from "text" column in lowercase
    message = re.sub(r'http[s]?://\S+', 'linkToken', message) # turns links into linkTokens
    message = re.sub(r'\+?\d[\d\s\-\(\)]{5,}\d', 'phoneNumberToken', message) # turns phone numbers into a phoneNumberToken \+? checks for + in the beginning, \d checks if starts with number, [\d\s\-\(\)], checks if continituous with \digit(number)\space\-\(\) so it would be good for 12345678 123-456-7890 (123) 456 7890 +49 157 1234567 and so on, {7,} checks if it's at least 7 digits long, \d checks if ends in digit
    message = message.split() #turns string into a list that consists out of each words in this very message
    message = [stemmer.stem(word) for word in message] #stemming it
    message_stripped = [word for word in [word.strip(",.;:-_/'?!") for word in message] if word != '' and (len(word) > 1 and word != "2")] #removes punctuation and words that are one character long
    message_strippedSet = set(message_stripped) #removes duplicate words(classical NB only cares if word appeared in the message, not how many times it appeared)


    if(row["status"] == False): #checks if this SMS is spam, False = spam, True = ham
        for i in message_strippedSet:#goes trough every word
            if i in spamWordsFrq: #checks if this word is already a key in dictionary
                spamWordsFrq[i] = spamWordsFrq[i] + 1 #if so increment the value of this word(we've seen it once more)
            else:
                spamWordsFrq[i] = 1 #if no then create key with this word that corresponds to one(we saw this word for the first time)
    else:
        for i in message_strippedSet: #does the same for ham
            if i in hamWordsFrq:
                hamWordsFrq[i] = hamWordsFrq[i] + 1
            else:
                hamWordsFrq[i] = 1
                

    for i in message_strippedSet: #does the same but for all messages without diffentiating between ham and spam
        if i in numberOfEachWord:
            numberOfEachWord[i] = numberOfEachWord[i]+1
        else:
            numberOfEachWord[i] = 1

#Now that it's done, there's not much left to do, just create parameters and P(y|x) using formulas you can find in README

#both phi's are gonna be dictionaries with key => word, value => percentage.
phij1 = {} #Instead of taking top 1000 words from internet and then estimating their probability of being in the spam I take words that appear in the messages.
phij0 = {}

#I'm not creating phi_y because I don't want the classifier to be biased, instead just setting it to 0.5. Although it can be a good idea because there are generally more ham messages than spam ones.
#It can be created the following way:

#AllSpamMessages = 0 
#for indx, row in SMS.iterrows(): #calculates how many are right
#    if row["status"] == False:
#        AllSpamMessages += 1

#phi_y = AllSpamMessages / len(SMS)


#first need to calculate how many messages are spam/ham
AllHamMessages = 0
AllSpamMessages = 0
for indx, row in SMS.iterrows(): #calculates how many are right
    if row["status"] == True:
        AllHamMessages += 1
    else:
        AllSpamMessages += 1




for key, value in spamWordsFrq.items(): # calculates phi1, spamWordsFrq is a dictionary that was created in the beginning and it contains words as keys and how many times they appeared in spam sms(only once) as values, 
    phij1[key] = (value + 1) / (AllSpamMessages + 2) #Calculates in how many messages word appeared divided by how many spam messages are there

for key, value in numberOfEachWord.items(): #spamWordsFrq probably doesn't include all words that appeared ham messages, so I look trough all words
    if(key in phij1): #if this word is already in phij1 move on
        continue
    else:
        phij1[key] = 1 / (AllSpamMessages + 2) #if not say it appeared in spam once (Laplace smoothing) and calculate

for key, value in hamWordsFrq.items():
    phij0[key] = (value + 1) / (AllHamMessages + 2) #does the same for phij0
    
for key, value in numberOfEachWord.items(): #just as with spam messages not being in ham, some ham words might not be in spam
    if(key in phij0):
        continue
    else:
        phij0[key] = 1 / (AllHamMessages + 2)






#building the function to determine P(y|x), so the function that tells if the message is spam or not.
#formula for this is also in README

def determineSpam(message, phi_y = 0.5, outputIn = "probability"): # outputIn can be either "probability" and return how likey message is spam(in percent) or "IsSpam" and return True or False where True is spam

    
    B = 0.0   
    A = 0.0

    probabilityY1X = 0.0 #final probability

    #repeating the same process for the new message classifier is looking at

    message = message.lower()
    message = re.sub(r'http[s]?://\S+', 'linkToken', message) 
    message = re.sub(r'\+?\d[\d\s\-\(\)]{5,}\d', 'phoneNumberToken', message)
    message = message.split()
    message = [stemmer.stem(word) for word in message]
    message_stripped = [word for word in [word.strip(",.;:-_/'?!") for word in message] if word != '' and (len(word) > 1 and word != "2")]
    message_strippedSet = set(message_stripped) #reminder that we don't need word dublicates because we only check if j-th word is in the message, not how many times it appeared


    # summation trough n words for A
    sumA = 0.0
    for word, frq in numberOfEachWord.items(): #loops trough every word in the dictionary
        xj = 0.0
        if(word in message_strippedSet): #checks if j-th word appears in the message
            xj = 1.0
        sumA += (xj * math.log(phij1[word])) + ((1-xj)*(math.log(1 - phij1[word]))) #just uses formula

    # summation trough n words for B
    sumB = 0.0 #does exactly the same but for B by replacing phij1 with phij0
    for word, frq in numberOfEachWord.items():
        xj = 0.0
        if(word in message_strippedSet):
            xj = 1.0
        sumB += (xj * math.log(phij0[word])) + ((1-xj)*(math.log(1 - phij0[word])))

    A = math.log(phi_y) + sumA #following formula
    B = math.log(1 - phi_y) + sumB

    probabilityY1X = 1 / (1 + math.e**(B-A)) # what is log of P(y=1|X) equal to

    if(outputIn == "probability"): #returns values depending on what is asked
        return probabilityY1X
    elif(outputIn == "IsSpam"):
        if(probabilityY1X >= 0.5):
            return "spam"
        else:
            return "ham"


#Now that the classifier is done, it's time to check how good it is.
#For this I'll use the remaining 700 messages or so and see how many of them it determines correctly. 

SMSTest = pd.read_csv("spamCleaned.csv", encoding = "latin-1", skiprows = range(1, 5001)) #takes messages after 5000 to check how good the classifier is

HamMessagesGuessedRight = 0
SpamMessagesGuessedRight = 0 #self explanatory

for indx, row in SMSTest.iterrows(): # look trough every message
    messageToProve = row["text"]
    if determineSpam(messageToProve, 0.5, "IsSpam") == "ham" and row["status"] == True: #if algorithm guesses ham right increment the value for ham
        HamMessagesGuessedRight += 1
    elif determineSpam(messageToProve, 0.5, "IsSpam") == "spam" and row["status"] == False: # same for spam
        SpamMessagesGuessedRight += 1
    #else:
    #    print(indx, determineSpam(messageToProve, 0.5, "probability"), messageToProve) #optional for checking how much percent it gave to messages that it guessed incorrectly

AllHamMessages = 0
AllSpamMessages = 0
for indx, row in SMSTest.iterrows(): #calculates how many are right
    if row["status"] == True:
        AllHamMessages += 1
    else:
        AllSpamMessages += 1

print("\n" * 5)
print("RESULTS: ", "\n")

print("Overall there are ", AllSpamMessages, "spam messages and ", AllHamMessages, "ham messages")
print("Algorithm guesses ", SpamMessagesGuessedRight, "spam messages and ", HamMessagesGuessedRight, "ham messages right")


#So in the end it gave pretty good results. It guesses 99.8% ham messages and 85.1% of spam messages correctly.
#What can further improve this model is probably more thorough proccessing of messages. Some new tokens can be introduced and removing numbers inside words could help.
