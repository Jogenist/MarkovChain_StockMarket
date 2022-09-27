# MarkovChain_StockMarket

The idea of this project is to predict the future behavior of stocks on the stock market by a pre-trained Markov chain. 
The Markov chains theory is a method of making quantitative analysis about the situation in which the system transfers from one state to another, 
hence predicting future tendencies. This provides a basis for making strategic analysis.

The implemented script loads historic stock market data via the yfinance package. After feature extraction the markov chain is trained and evaluated.
The markov chain tries to predict whether or not a stock is rising in its value the following day.
Depending on the chosen stocks for training and evaluation an accuracy of up to 56% can be achieved. (better than guessing)
