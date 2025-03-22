"""
Feature engineering module for creating technical indicators and financial features.
"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("feature_engineering")

# Technical indicators implementation to replace TA-Lib
class TechnicalIndicators:
    @staticmethod
    def SMA(close, timeperiod=14):
        """Simple Moving Average"""
        return close.rolling(window=timeperiod).mean()
    
    @staticmethod
    def EMA(close, timeperiod=14):
        """Exponential Moving Average"""
        return close.ewm(span=timeperiod, adjust=False).mean()
    
    @staticmethod
    def RSI(close, timeperiod=14):
        """Relative Strength Index"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=timeperiod).mean()
        avg_loss = loss.rolling(window=timeperiod).mean()
        
        # Handle division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rs = rs.fillna(0)
        
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
        """Bollinger Bands"""
        middle = close.rolling(window=timeperiod).mean()
        std_dev = close.rolling(window=timeperiod).std()
        
        upper = middle + (std_dev * nbdevup)
        lower = middle - (std_dev * nbdevdn)
        
        return upper, middle, lower
    
    @staticmethod
    def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
        """Moving Average Convergence Divergence"""
        fast_ema = close.ewm(span=fastperiod, adjust=False).mean()
        slow_ema = close.ewm(span=slowperiod, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        """Average True Range"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=timeperiod).mean()
        
        return atr
    
    @staticmethod
    def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=fastk_period).min()
        highest_high = high.rolling(window=fastk_period).max()
        
        # Handle division by zero
        denom = highest_high - lowest_low
        denom = denom.replace(0, np.nan)
        
        fastk = 100 * ((close - lowest_low) / denom)
        fastk = fastk.fillna(50)  # Default to middle value when undefined
        
        slowk = fastk.rolling(window=slowk_period).mean()
        slowd = slowk.rolling(window=slowd_period).mean()
        
        return slowk, slowd
    
    @staticmethod
    def OBV(close, volume):
        """On Balance Volume"""
        obv = pd.Series(index=close.index)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    @staticmethod
    def SUPERTREND(high, low, close, period=14, multiplier=3.0):
        """
        SuperTrend indicator
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            period (int): ATR period
            multiplier (float): ATR multiplier
        
        Returns:
            pd.Series: SuperTrend line
            pd.Series: SuperTrend direction (1 for bullish, -1 for bearish)
        """
        # Calculate ATR
        atr = TechnicalIndicators.ATR(high, low, close, timeperiod=period)
        
        # Calculate Basic Upper and Lower Bands
        hl2 = (high + low) / 2
        
        # Initialize Supertrend columns
        supertrend = pd.Series(0.0, index=close.index)
        final_upperband = pd.Series(0.0, index=close.index)
        final_lowerband = pd.Series(0.0, index=close.index)
        supertrend_direction = pd.Series(1, index=close.index)
        
        # Initialize first values
        for i in range(period, len(close)):
            final_upperband.iloc[i] = hl2.iloc[i] + (multiplier * atr.iloc[i])
            final_lowerband.iloc[i] = hl2.iloc[i] - (multiplier * atr.iloc[i])
            
            # Check for supertrend direction and value
            if close.iloc[i] <= final_upperband.iloc[i]:
                supertrend_direction.iloc[i] = -1
            else:
                supertrend_direction.iloc[i] = 1
                
            if supertrend_direction.iloc[i] == 1:
                supertrend.iloc[i] = final_lowerband.iloc[i]
            else:
                supertrend.iloc[i] = final_upperband.iloc[i]
                
        return supertrend, supertrend_direction
    
    @staticmethod
    def ICHIMOKU(high, low, close, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
        """
        Ichimoku Cloud indicator
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            tenkan_period (int): Tenkan-sen (Conversion Line) period
            kijun_period (int): Kijun-sen (Base Line) period
            senkou_b_period (int): Senkou Span B period
            displacement (int): Displacement for Senkou Span
            
        Returns:
            dict: Ichimoku cloud components
        """
        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past tenkan_period
        tenkan_sen = pd.Series((high.rolling(window=tenkan_period).max() + 
                             low.rolling(window=tenkan_period).min()) / 2, index=close.index)
        
        # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for the past kijun_period
        kijun_sen = pd.Series((high.rolling(window=kijun_period).max() + 
                            low.rolling(window=kijun_period).min()) / 2, index=close.index)
        
        # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 (displaced forward by displacement)
        senkou_span_a = pd.Series((tenkan_sen + kijun_sen) / 2, index=close.index).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past senkou_b_period (displaced forward)
        senkou_span_b = pd.Series((high.rolling(window=senkou_b_period).max() + 
                                low.rolling(window=senkou_b_period).min()) / 2, 
                               index=close.index).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span): Current closing price (displaced backwards by displacement)
        chikou_span = pd.Series(close, index=close.index).shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def ADX(high, low, close, timeperiod=14):
        """
        Average Directional Index (ADX)
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            timeperiod (int): Period for calculations
            
        Returns:
            tuple: (ADX, +DI, -DI)
        """
        # Get ATR first
        atr = TechnicalIndicators.ATR(high, low, close, timeperiod=timeperiod)
        
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = low.diff().abs()
        
        # +DM
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        
        # -DM
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=timeperiod).sum() / atr.rolling(window=timeperiod).sum())
        minus_di = 100 * (minus_dm.rolling(window=timeperiod).sum() / atr.rolling(window=timeperiod).sum())
        
        # Calculate DX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).abs())
        
        # Calculate ADX
        adx = dx.rolling(window=timeperiod).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def PIVOT_POINTS(high, low, close, method='standard'):
        """
        Calculate Pivot Points
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            method (str): Pivot point calculation method ('standard', 'fibonacci', 'woodie', 'camarilla')
            
        Returns:
            dict: Pivot point levels (P, S1, S2, S3, R1, R2, R3)
        """
        result = {}
        
        # Standard calculation uses previous day's high, low, and close
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        # Default initialization to avoid "possibly unbound" errors
        p = pd.Series(0.0, index=close.index)
        r1 = pd.Series(0.0, index=close.index)
        s1 = pd.Series(0.0, index=close.index)
        r2 = pd.Series(0.0, index=close.index)
        s2 = pd.Series(0.0, index=close.index)
        r3 = pd.Series(0.0, index=close.index)
        s3 = pd.Series(0.0, index=close.index)
        
        if method == 'standard':
            p = (prev_high + prev_low + prev_close) / 3
            r1 = (2 * p) - prev_low
            s1 = (2 * p) - prev_high
            r2 = p + (prev_high - prev_low)
            s2 = p - (prev_high - prev_low)
            r3 = p + 2 * (prev_high - prev_low)
            s3 = p - 2 * (prev_high - prev_low)
            
        elif method == 'fibonacci':
            p = (prev_high + prev_low + prev_close) / 3
            r1 = p + 0.382 * (prev_high - prev_low)
            s1 = p - 0.382 * (prev_high - prev_low)
            r2 = p + 0.618 * (prev_high - prev_low)
            s2 = p - 0.618 * (prev_high - prev_low)
            r3 = p + 1.0 * (prev_high - prev_low)
            s3 = p - 1.0 * (prev_high - prev_low)
            
        elif method == 'woodie':
            p = (prev_high + prev_low + 2 * close) / 4  # Uses current close
            r1 = (2 * p) - prev_low
            s1 = (2 * p) - prev_high
            r2 = p + (prev_high - prev_low)
            s2 = p - (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            
        elif method == 'camarilla':
            p = (prev_high + prev_low + prev_close) / 3
            r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
            s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
            r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
            s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
            r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
            s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
        
        result['pivot'] = p
        result['r1'] = r1
        result['s1'] = s1
        result['r2'] = r2
        result['s2'] = s2
        result['r3'] = r3
        result['s3'] = s3
        
        return result
    
    @staticmethod
    def VOLUME_PROFILE(high, low, close, volume, divisions=10):
        """
        Calculate Volume Profile (simplified version)
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            volume (pd.Series): Volume data
            divisions (int): Number of price divisions
            
        Returns:
            pd.DataFrame: Volume profile data
        """
        # Find price range
        price_range = high.max() - low.min()
        division_size = price_range / divisions
        
        # Create price levels
        price_levels = [low.min() + (i * division_size) for i in range(divisions+1)]
        
        # Initialize volume at each price level
        volume_at_price = {level: 0 for level in price_levels[:-1]}
        
        # Assign volume to price levels
        for i in range(len(close)):
            for level_idx in range(len(price_levels)-1):
                lower = price_levels[level_idx]
                upper = price_levels[level_idx+1]
                
                # If price traded within this level
                if low.iloc[i] <= upper and high.iloc[i] >= lower:
                    # Roughly estimate the portion of volume at this price level
                    # This is a simplification - in reality would need more precise calculation
                    overlap = min(high.iloc[i], upper) - max(low.iloc[i], lower)
                    price_range_candle = high.iloc[i] - low.iloc[i]
                    
                    if price_range_candle > 0:
                        volume_portion = overlap / price_range_candle
                    else:
                        volume_portion = 1 if close.iloc[i] >= lower and close.iloc[i] < upper else 0
                        
                    volume_at_price[lower] += volume.iloc[i] * volume_portion
        
        # Convert to DataFrame
        volume_profile = pd.DataFrame({
            'price_level': list(volume_at_price.keys()),
            'volume': list(volume_at_price.values())
        })
        
        # Find POC (Point of Control) - level with highest volume
        poc_level = volume_profile.loc[volume_profile['volume'].idxmax(), 'price_level']
        
        return volume_profile, poc_level

class FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer with scalers for normalization."""
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(feature_range=(-1, 1))
        }
        self.fitted = False

    def add_basic_price_features(self, df):
        """
        Add basic price derived features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added features
        """
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # Price differences
        df_features['price_change'] = df_features['close'] - df_features['open']
        df_features['price_change_pct'] = df_features['price_change'] / df_features['open']
        
        # Range features
        df_features['high_low_range'] = df_features['high'] - df_features['low']
        df_features['high_low_range_pct'] = df_features['high_low_range'] / df_features['open']
        
        # Volume features
        df_features['volume_change'] = df_features['volume'].pct_change()
        df_features['volume_ma'] = df_features['volume'].rolling(window=20).mean()
        df_features['relative_volume'] = df_features['volume'] / df_features['volume_ma']
        
        # Log returns
        df_features['log_return'] = np.log(df_features['close'] / df_features['close'].shift(1))
        
        # Candlestick pattern features
        df_features['body_size'] = abs(df_features['close'] - df_features['open'])
        df_features['upper_shadow'] = df_features['high'] - df_features[['open', 'close']].max(axis=1)
        df_features['lower_shadow'] = df_features[['open', 'close']].min(axis=1) - df_features['low']
        
        # Calculate if candle is bullish or bearish
        df_features['is_bullish'] = (df_features['close'] > df_features['open']).astype(int)
        
        # Calculate price position relative to candle
        df_features['price_position'] = (df_features['close'] - df_features['low']) / (df_features['high'] - df_features['low'])
        
        # Drop rows with NaN values created by rolling calculations
        df_features.dropna(inplace=True)
        
        return df_features

    def add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        # Make a copy to avoid modifying the original
        df_indicators = df.copy()
        
        # Check if we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_indicators.columns for col in required_columns):
            logger.error(f"DataFrame is missing required columns: {required_columns}")
            return df
        
        try:
            # Create the technical indicators instance
            ti = TechnicalIndicators()
            
            # RSI
            window = config.TECHNICAL_INDICATORS['rsi']['window']
            df_indicators[f'rsi_{window}'] = ti.RSI(df_indicators['close'], timeperiod=window)
            
            # EMA (multiple windows)
            for window in config.TECHNICAL_INDICATORS['ema']['windows']:
                df_indicators[f'ema_{window}'] = ti.EMA(df_indicators['close'], timeperiod=window)
            
            # Calculate EMA crossovers
            if len(config.TECHNICAL_INDICATORS['ema']['windows']) > 1:
                windows = config.TECHNICAL_INDICATORS['ema']['windows']
                for i in range(len(windows)):
                    for j in range(i+1, len(windows)):
                        shorter = windows[i]
                        longer = windows[j]
                        df_indicators[f'ema_{shorter}_{longer}_diff'] = (
                            df_indicators[f'ema_{shorter}'] - df_indicators[f'ema_{longer}']
                        )
                        df_indicators[f'ema_{shorter}_{longer}_cross'] = np.sign(
                            df_indicators[f'ema_{shorter}_{longer}_diff']
                        ).diff().fillna(0)
            
            # MACD
            macd_config = config.TECHNICAL_INDICATORS['macd']
            macd, macdsignal, macdhist = ti.MACD(
                df_indicators['close'],
                fastperiod=macd_config['fast'],
                slowperiod=macd_config['slow'],
                signalperiod=macd_config['signal']
            )
            df_indicators['macd'] = macd
            df_indicators['macd_signal'] = macdsignal
            df_indicators['macd_hist'] = macdhist
            df_indicators['macd_cross'] = np.sign(macd - macdsignal).diff().fillna(0)
            
            # Bollinger Bands
            bbands_config = config.TECHNICAL_INDICATORS['bbands']
            upperband, middleband, lowerband = ti.BBANDS(
                df_indicators['close'],
                timeperiod=bbands_config['window'],
                nbdevup=bbands_config['std_dev'],
                nbdevdn=bbands_config['std_dev']
            )
            df_indicators['bb_upper'] = upperband
            df_indicators['bb_middle'] = middleband
            df_indicators['bb_lower'] = lowerband
            
            # BB position (where current price is relative to the bands)
            df_indicators['bb_position'] = (df_indicators['close'] - df_indicators['bb_lower']) / (
                df_indicators['bb_upper'] - df_indicators['bb_lower']
            )
            
            # BB width
            df_indicators['bb_width'] = (df_indicators['bb_upper'] - df_indicators['bb_lower']) / df_indicators['bb_middle']
            
            # ATR (Average True Range)
            atr_window = config.TECHNICAL_INDICATORS['atr']['window']
            df_indicators[f'atr_{atr_window}'] = ti.ATR(
                df_indicators['high'],
                df_indicators['low'],
                df_indicators['close'],
                timeperiod=atr_window
            )
            
            # Normalized ATR
            df_indicators[f'atr_{atr_window}_normalized'] = df_indicators[f'atr_{atr_window}'] / df_indicators['close']
            
            # VWAP (Volume Weighted Average Price)
            vwap_window = config.TECHNICAL_INDICATORS['vwap']['window']
            df_indicators['vwap'] = (
                (df_indicators['volume'] * df_indicators['close']).rolling(window=vwap_window).sum() / 
                df_indicators['volume'].rolling(window=vwap_window).sum()
            )
            
            # Stochastic Oscillator
            slowk, slowd = ti.STOCH(
                df_indicators['high'],
                df_indicators['low'],
                df_indicators['close'],
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )
            df_indicators['stoch_k'] = slowk
            df_indicators['stoch_d'] = slowd
            df_indicators['stoch_cross'] = np.sign(slowk - slowd).diff().fillna(0)
            
            # OBV (On Balance Volume)
            df_indicators['obv'] = ti.OBV(df_indicators['close'], df_indicators['volume'])
            
            # --- Begin Advanced Technical Indicators ---
            
            # SuperTrend
            if 'supertrend' in config.TECHNICAL_INDICATORS:
                st_config = config.TECHNICAL_INDICATORS['supertrend']
                supertrend, direction = ti.SUPERTREND(
                    df_indicators['high'],
                    df_indicators['low'],
                    df_indicators['close'],
                    period=st_config['period'],
                    multiplier=st_config['multiplier']
                )
                df_indicators['supertrend'] = supertrend
                df_indicators['supertrend_direction'] = direction
                # Calculate if price is above or below supertrend
                df_indicators['supertrend_position'] = (df_indicators['close'] > df_indicators['supertrend']).astype(int)
            
            # Ichimoku Cloud
            if 'ichimoku' in config.TECHNICAL_INDICATORS:
                ichi_config = config.TECHNICAL_INDICATORS['ichimoku']
                ichimoku = ti.ICHIMOKU(
                    df_indicators['high'],
                    df_indicators['low'],
                    df_indicators['close'],
                    tenkan_period=ichi_config['tenkan_period'],
                    kijun_period=ichi_config['kijun_period'],
                    senkou_b_period=ichi_config['senkou_b_period'],
                    displacement=ichi_config['displacement']
                )
                
                df_indicators['ichimoku_tenkan'] = ichimoku['tenkan_sen']
                df_indicators['ichimoku_kijun'] = ichimoku['kijun_sen']
                df_indicators['ichimoku_senkou_a'] = ichimoku['senkou_span_a']
                df_indicators['ichimoku_senkou_b'] = ichimoku['senkou_span_b']
                df_indicators['ichimoku_chikou'] = ichimoku['chikou_span']
                
                # Calculate cloud strength
                df_indicators['ichimoku_cloud_strength'] = ichimoku['senkou_span_a'] - ichimoku['senkou_span_b']
                
                # Tenkan/Kijun Cross
                df_indicators['ichimoku_tk_cross'] = np.sign(
                    ichimoku['tenkan_sen'] - ichimoku['kijun_sen']
                ).diff().fillna(0)
                
                # Price relative to cloud
                df_indicators['ichimoku_cloud_position'] = np.where(
                    df_indicators['close'] > df_indicators['ichimoku_senkou_a'],
                    np.where(
                        df_indicators['close'] > df_indicators['ichimoku_senkou_b'],
                        2,  # Above both spans (strong bullish)
                        1   # Between spans
                    ),
                    np.where(
                        df_indicators['close'] < df_indicators['ichimoku_senkou_b'],
                        -2, # Below both spans (strong bearish)
                        -1  # Between spans
                    )
                )
            
            # ADX (Average Directional Index)
            if 'adx' in config.TECHNICAL_INDICATORS:
                adx_window = config.TECHNICAL_INDICATORS['adx']['window']
                adx, plus_di, minus_di = ti.ADX(
                    df_indicators['high'],
                    df_indicators['low'],
                    df_indicators['close'],
                    timeperiod=adx_window
                )
                df_indicators['adx'] = adx
                df_indicators['plus_di'] = plus_di
                df_indicators['minus_di'] = minus_di
                
                # Trend strength indicator (ADX above 25 is considered strong trend)
                df_indicators['trend_strength'] = (adx > 25).astype(int)
                
                # Trend direction indicator
                df_indicators['adx_trend_direction'] = np.sign(plus_di - minus_di)
                
                # DI Crossovers
                df_indicators['di_cross'] = np.sign(plus_di - minus_di).diff().fillna(0)
            
            # Pivot Points
            if 'pivot_points' in config.TECHNICAL_INDICATORS:
                pp_config = config.TECHNICAL_INDICATORS['pivot_points']
                
                # Check if we have enough data for daily pivots
                if len(df_indicators) >= 2:  # Need at least previous day's data
                    pivot_points = ti.PIVOT_POINTS(
                        df_indicators['high'],
                        df_indicators['low'],
                        df_indicators['close'],
                        method=pp_config['method']
                    )
                    
                    # Add pivot points to dataframe
                    df_indicators['pivot'] = pivot_points['pivot']
                    df_indicators['pivot_r1'] = pivot_points['r1']
                    df_indicators['pivot_s1'] = pivot_points['s1']
                    df_indicators['pivot_r2'] = pivot_points['r2']
                    df_indicators['pivot_s2'] = pivot_points['s2']
                    df_indicators['pivot_r3'] = pivot_points['r3']
                    df_indicators['pivot_s3'] = pivot_points['s3']
                    
                    # Calculate distance from current price to nearest pivot level
                    pivot_levels = [
                        df_indicators['pivot'],
                        df_indicators['pivot_r1'],
                        df_indicators['pivot_s1'],
                        df_indicators['pivot_r2'],
                        df_indicators['pivot_s2'],
                        df_indicators['pivot_r3'],
                        df_indicators['pivot_s3']
                    ]
                    
                    # Find closest pivot level
                    closest_distance = float('inf')
                    for level in pivot_levels:
                        distance = abs(df_indicators['close'] - level)
                        closest_distance = np.minimum(closest_distance, distance)
                    
                    df_indicators['pivot_distance'] = closest_distance / df_indicators['close']  # Normalized distance
            
            # Volume Profile - This is computationally expensive, so only calculate for a reasonable subset
            if 'volume_profile' in config.TECHNICAL_INDICATORS and len(df_indicators) <= 1000:
                vp_config = config.TECHNICAL_INDICATORS['volume_profile']
                
                try:
                    # Calculate Volume Profile for the entire dataset
                    volume_profile, poc_level = ti.VOLUME_PROFILE(
                        df_indicators['high'],
                        df_indicators['low'],
                        df_indicators['close'],
                        df_indicators['volume'],
                        divisions=vp_config['divisions']
                    )
                    
                    # Add Point of Control to dataframe
                    df_indicators['volume_poc'] = poc_level
                    
                    # Calculate price distance from POC
                    df_indicators['poc_distance'] = (df_indicators['close'] - poc_level) / df_indicators['close']
                except Exception as e:
                    logger.warning(f"Error calculating Volume Profile: {e}")
            
            # --- End Advanced Technical Indicators ---
            
            # Drop rows with NaN values created by technical indicators
            df_indicators.dropna(inplace=True)
            
            return df_indicators
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    def add_target_labels(self, df, window=config.PREDICTION_WINDOW, 
                        threshold=config.PRICE_MOVEMENT_THRESHOLD,
                        pnl_threshold=config.TARGET_PNL_THRESHOLD):
        """
        Add target labels for supervised learning.
        
        Args:
            df (pd.DataFrame): DataFrame with feature data
            window (int): Number of future candles to look ahead
            threshold (float): Price movement threshold for labeling
            pnl_threshold (float): Alternative PNL threshold in USDT
            
        Returns:
            pd.DataFrame: DataFrame with target labels
        """
        # Make a copy to avoid modifying the original
        df_labeled = df.copy()
        
        try:
            # Calculate future returns
            future_close = df_labeled['close'].shift(-window)
            future_return = (future_close - df_labeled['close']) / df_labeled['close']
            
            # Calculate future PNL (in USDT) for 1 ETH
            future_pnl = future_close - df_labeled['close']
            
            # Create target labels based on percentage movement
            df_labeled['target_pct'] = future_return
            
            # Create target labels based on movement direction with threshold
            df_labeled['target_class'] = np.where(
                future_return > threshold, 2,  # LONG
                np.where(
                    future_return < -threshold, 0,  # SHORT
                    1  # NEUTRAL
                )
            )
            
            # Alternative labeling based on PNL threshold
            df_labeled['target_pnl'] = future_pnl
            df_labeled['target_pnl_class'] = np.where(
                future_pnl > pnl_threshold, 2,  # LONG
                np.where(
                    future_pnl < -pnl_threshold, 0,  # SHORT
                    1  # NEUTRAL
                )
            )
            
            # Remove the last 'window' rows as they don't have targets
            df_labeled = df_labeled.iloc[:-window]
            
            return df_labeled
            
        except Exception as e:
            logger.error(f"Error adding target labels: {e}")
            return df

    def normalize_features(self, df, method='minmax', fit=True):
        """
        Normalize numerical features using various scaling methods.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            method (str): Scaling method ('standard' or 'minmax')
            fit (bool): Whether to fit the scaler or use pre-fitted
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        # Make a copy to avoid modifying the original
        df_norm = df.copy()
        
        # Exclude non-numeric columns and target columns
        target_columns = [col for col in df_norm.columns if col.startswith('target_')]
        exclude_columns = target_columns + ['open_time', 'close_time']
        
        # Get numeric columns excluding the ones we want to preserve
        numeric_columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        scale_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        try:
            # Extract features to normalize
            features = df_norm[scale_columns].values
            
            # Kiểm tra xem features có dữ liệu không trước khi chuẩn hóa
            if features.shape[0] == 0:
                logger.warning("Empty dataset received for normalization. Skipping normalization.")
                return df_norm
                
            # Normalize features
            scaler = self.scalers.get(method, self.scalers['minmax'])
            
            if fit:
                normalized_features = scaler.fit_transform(features)
                self.fitted = True
            else:
                if not self.fitted:
                    logger.warning("Scaler not fitted yet. Fitting now...")
                    normalized_features = scaler.fit_transform(features)
                    self.fitted = True
                else:
                    normalized_features = scaler.transform(features)
            
            # Replace original features with normalized ones
            df_norm[scale_columns] = normalized_features
            
            return df_norm
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return df

    def preprocess_data(self, df, add_basic=True, add_technical=True, 
                      add_labels=True, normalize=True):
        """
        Full data preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Raw OHLCV DataFrame
            add_basic (bool): Whether to add basic price features
            add_technical (bool): Whether to add technical indicators
            add_labels (bool): Whether to add target labels
            normalize (bool): Whether to normalize features
            
        Returns:
            pd.DataFrame: Fully processed DataFrame ready for modeling
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Process data through pipeline
        try:
            processed_df = df.copy()
            
            if add_basic:
                logger.info("Adding basic price features")
                processed_df = self.add_basic_price_features(processed_df)
                
            if add_technical:
                logger.info("Adding technical indicators")
                processed_df = self.add_technical_indicators(processed_df)
                
            if add_labels:
                logger.info("Adding target labels")
                processed_df = self.add_target_labels(processed_df)
                
            if normalize:
                logger.info("Normalizing features")
                processed_df = self.normalize_features(processed_df)
            
            # Calculate and log dataset statistics
            if not processed_df.empty:
                logger.info(f"Processed dataset has {len(processed_df)} samples")
                
                if 'target_class' in processed_df.columns:
                    class_counts = processed_df['target_class'].value_counts()
                    logger.info(f"Class distribution: {class_counts.to_dict()}")
                    
                logger.info("Feature engineering completed successfully")
            else:
                logger.warning("Processed dataset is empty!")
                
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing pipeline: {e}")
            return df

    def create_sequences(self, df, target_col='target_class', 
                        seq_length=config.SEQUENCE_LENGTH):
        """
        Create sequences for time series models like LSTM and Transformer.
        
        Args:
            df (pd.DataFrame): Processed feature DataFrame
            target_col (str): Target column name
            seq_length (int): Sequence length (number of timesteps)
            
        Returns:
            tuple: (X_sequences, y_targets) for model training
        """
        # Check if target column exists
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in DataFrame")
            return None, None
            
        try:
            # Get feature columns (excluding target columns)
            feature_cols = [col for col in df.columns if not col.startswith('target_')]
            
            # Initialize lists to store sequences and targets
            X_sequences = []
            y_targets = []
            
            # Create sequences
            for i in range(len(df) - seq_length):
                X_sequences.append(df[feature_cols].iloc[i:i+seq_length].values)
                y_targets.append(df[target_col].iloc[i+seq_length-1])
            
            # Convert to numpy arrays
            X_sequences = np.array(X_sequences)
            y_targets = np.array(y_targets)
            
            logger.info(f"Created {len(X_sequences)} sequences of length {seq_length}")
            logger.info(f"Sequence shape: {X_sequences.shape}, Target shape: {y_targets.shape}")
            
            return X_sequences, y_targets
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return None, None

    def create_image_data(self, df, seq_length=config.SEQUENCE_LENGTH, target_col='target_class'):
        """
        Create image-like data for CNN models based on OHLCV data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            seq_length (int): Number of candles per image
            target_col (str): Target column name, can be None for prediction
            
        Returns:
            tuple: (X_images, y_targets) for CNN model training
        """
        try:
            # Check if target column exists, for prediction we may not have targets
            if target_col in df.columns:
                # Create sequences with targets for training
                feature_sequences, targets = self.create_sequences(df, target_col=target_col, seq_length=seq_length)
                
                if feature_sequences is None:
                    return None, None
            else:
                # For prediction, create sequences without expecting target column
                # Get feature columns (excluding any target columns if they exist)
                feature_cols = [col for col in df.columns if not col.startswith('target_')]
                
                # Initialize lists to store sequences
                X_sequences = []
                
                # Create sequences
                for i in range(len(df) - seq_length):
                    X_sequences.append(df[feature_cols].iloc[i:i+seq_length].values)
                
                # Convert to numpy arrays
                feature_sequences = np.array(X_sequences)
                targets = np.zeros(len(feature_sequences))  # Dummy targets
                
                if len(feature_sequences) == 0:
                    logger.error("No sequences could be created - not enough data")
                    return None, None
                
                logger.info(f"Created {len(feature_sequences)} sequences for prediction")
                
            # Extract only OHLCV data for the candlestick visualization
            # This is a simplified version - actual candlestick images would be more complex
            n_sequences = feature_sequences.shape[0]
            
            # Create a 5-channel image-like data structure (OHLCV)
            # Shape: (n_samples, seq_length, 5, 1) for image format
            X_images = np.zeros((n_sequences, seq_length, 5, 1))
            
            # Fill in the image data with OHLCV values
            # Assuming first 5 columns are Open, High, Low, Close, Volume
            for i in range(n_sequences):
                for j in range(5):  # 5 channels for OHLCV
                    X_images[i, :, j, 0] = feature_sequences[i, :, j]
            
            logger.info(f"Created {len(X_images)} image-like data of shape {X_images.shape}")
            
            return X_images, targets
            
        except Exception as e:
            logger.error(f"Error creating image data: {e}")
            return None, None
