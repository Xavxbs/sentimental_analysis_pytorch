这里使用lstm实现了一个简单的twitter内容情感分析。
这只是一个baseline，没有经过dropout正则化以及各种改进的方式。
Dataset was download from http://help.sentiment140.com/for-students
The data is a CSV with emoticons removed. Data file format has 6 fields:
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)(trainingset only contains 2 kinds of labels)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is cool)

