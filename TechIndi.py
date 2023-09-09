import pandas as pd
import numpy as np

class TechnicalIndicators:
    
    def moving_average(df, n):
        """Calculate the moving average for the given data.
        :param df: pandas.DataFrame
        :param n: window
        :return: pandas.DataFrame
        """
        MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_{}'.format(n))
        
        return MA
    
    
    def exponential_moving_average(df, n):
        """
        :param df: pandas.DataFrame
        :param n: window of data to take moving exponent mean
        :return: pandas.DataFrame
        """
        EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
        return EMA
    
    def commodity_channel_index(df, n):
        """Calculate Commodity Channel Index for given data.
        :param df: pandas.DataFrame
        :param n: data window
        :return: pandas.DataFrame
        """
        PP = (df['High'] + df['Low'] + df['Close']) / 3
        CCI = pd.Series((PP - PP.rolling(n, min_periods=n).mean()) / PP.rolling(n, min_periods=n).std(),
                        name='CCI_' + str(n))
        return CCI
    
    def momentum(df, n):
        """
        :param df: pandas.DataFrame 
        :param n: data window
        :return: pandas.DataFrame
        """
        M = pd.Series(df['Close'].diff(n), name='Momentum_' + str(n))
        return M
    
    def stochastic_oscillator_k(df):
        """Calculate stochastic oscillator %K for given data.
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
        return SOk
    
    
    def stochastic_oscillator_d(df, n):
        """Calculate stochastic oscillator %D for given data.
        :param df: pandas.DataFrame
        :param n: data window
        :return: pandas.DataFrame
        """
        SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
        SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SO%d_' + str(n))
        return SOd
    
    def mass_index(df, n):
        """Calculate the Mass Index for given data.
        
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        Range = df['High'] - df['Low']
        EX1 = Range.ewm(span=9, min_periods=9).mean()
        EX2 = EX1.ewm(span=9, min_periods=9).mean()
        Mass = EX1 / EX2
        MassI = pd.Series(Mass.rolling(n).sum(), name='Mass Index')
        return MassI
    
    def force_index(df, n):
        """Calculate Force Index for given data.
        
        :param df: pandas.DataFrame
        :param n: data window
        :return: pandas.DataFrame
        """
        F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name='Force_' + str(n))
        return F
