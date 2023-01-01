updated version of solving_ipda_with_dqn_2.

<h1>Solving the IPD-Algorithm using deep q learning</h1>

In this project i try to train a DQN to navigate through the forex market and be a profitable trading Algorithm.
This is not the first project that tries to create a trading algorithm, but it is a very different approach from what is commonly used to do so.
Most attempts in RL based trading focus on daily candlesticks, many do not include brokerage fees.
They use some last 20 price bars and some indicators and try to squeeze it through an lstm expecting good results. 

So why is everyone talking about gpt models and image generation but all the trading models are not used anywhere after training?
It is because they do not work outside of the training data!

To create a NN that can trade the market, you have to know how the markets work. 
For learning about the market i recommend the free Youtube library of Michael J. Huddleston (aka ICT).<br>
<b>Everything in this Project is based on the research done by ICT!</b>

Here are just a few things that make trading NNs fail: 
<ol>
  <li> 20 bars say nearly nothing about price movement</li> 
  <li> Price does not care about indicators</li>
  <li> With daily candlesticks there is not enough data to train a network without overfitting</li>
  <li> Higher timeframe movement is influenced on fundamentals and this is not included in candlestick based training</li>
  <li> Most Developers do know much about Deep learning and Neural Networks but do not know about the markets and do not know how to prepare candlestick data that a Neural Network can learn from it. A NN will not learn from a price vector of 20-40 bars scaled to -1 and 1 because it has no information that can be learned. (data preporcessing)
</ol>
    

I will not remake the ICT mentorship in this but here are some main ideas about the market and how a NN can be used to learn it:
<ul>
  <li>Price is delivered by an algorithm, it is rule based and not random - <b>This can be learned by a neural network</b></li>
  <li>Intraday volatility is controlled to the Pip based on time and price</li>
  <li>A neural network should be able to determine the next draw on liquidity based on HTF charts and LTF charts and get fair value entries</li>
  <li>The input data for the neural network needs to have the information to solve #3</li>
  <li>The neural network needs to be able to process the input data in a way that it can extract the important information (so not use just rnn with a simple price vector)</li>
</ul>

<h3>How does this project implements the ideas listed above?</h3>
In this Project the DQN gets input data from multiple timeframes: m5, m15, h1, h4, d1 and its current position (1/-1 = long/short)
The number of candles that get pushed into the DQN is based on the IPDA-Data-ranges of 20days, 40days and 60days lookback + look-forward = 120 candlesticks.
This number is used on all timeframes.<br>
Candlestick data is encoded as a 2d array like the chart picture. Every candle is represented by 1 column in the picture and scaled down to a given max height of the picture (at time of writing this is set to 100). Every timeframe will be a picture of 120px width and 100px height.<br>

<ul>
  <li>Brokerage fees are included in training (15/100000 - this is higher than a good broker offers)</li>
  <li>Calculations are done on the 5min timeframe</li>
  <li>training data is about 20 years on 15 currency pairs, 7/15 are USD pairs</li>
</ul>  


tx_1: support for tf strategy for multiple gpus or tpus (kaggle or tpu research cloud), currently trained on kaggle at https://www.kaggle.com/bpwqsdd/ipda-vs-dqn-5

tx_2: single gpu, currently trained on my laptop
