"""
Module bộ lọc thị trường để phân tích điều kiện thị trường tổng thể
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import config

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("market_filter")

class MarketFilter:
    """
    Bộ lọc thị trường cơ bản để phân tích xu hướng thị trường và điều kiện giao dịch
    """
    def __init__(self, data_collector=None):
        """
        Khởi tạo bộ lọc thị trường
        
        Args:
            data_collector: Data collector để lấy dữ liệu từ Binance
        """
        self.data_collector = data_collector
        self.btc_data = None
        self.eth_data = None
        self.market_state = {
            'trend': 'UNKNOWN',
            'volatility': 'UNKNOWN',
            'correlation': 'UNKNOWN',
            'recommend': 'WAIT',
            'conditions': []
        }
        self.last_update = None
        
    def update(self, eth_data=None, force_update=False):
        """
        Cập nhật trạng thái thị trường
        
        Args:
            eth_data (pd.DataFrame): Dữ liệu ETH đã có sẵn (nếu có)
            force_update (bool): Bắt buộc cập nhật ngay cả khi mới cập nhật gần đây
            
        Returns:
            dict: Trạng thái thị trường
        """
        # Kiểm tra xem có cần cập nhật hay không
        now = datetime.now()
        if not force_update and self.last_update and (now - self.last_update) < timedelta(minutes=15):
            logger.info("Sử dụng kết quả phân tích thị trường đã lưu trong cache")
            return self.market_state
        
        # Lấy dữ liệu ETH nếu chưa có
        if eth_data is not None:
            self.eth_data = eth_data
        elif self.data_collector:
            try:
                # Lấy dữ liệu ETH cho timeframe 1 giờ (đủ dài để phân tích xu hướng)
                eth_data = self.data_collector.collect_historical_data(
                    symbol=config.SYMBOL,
                    timeframe="1h",
                    limit=500  # Khoảng 21 ngày
                )
                self.eth_data = eth_data
                logger.info(f"Đã lấy dữ liệu ETH 1h, kích thước: {len(eth_data)}")
            except Exception as e:
                logger.error(f"Lỗi khi lấy dữ liệu ETH: {e}")
                if self.eth_data is None:
                    # Nếu không có dữ liệu từ trước, không thể phân tích
                    return self.market_state
        
        # Lấy dữ liệu BTC để tính tương quan
        if self.data_collector:
            try:
                # Lấy dữ liệu BTC cho timeframe 1 giờ (đủ dài để phân tích xu hướng)
                btc_data = self.data_collector.collect_historical_data(
                    symbol="BTCUSDT",
                    timeframe="1h",
                    limit=500  # Khoảng 21 ngày
                )
                self.btc_data = btc_data
                logger.info(f"Đã lấy dữ liệu BTC 1h, kích thước: {len(btc_data)}")
            except Exception as e:
                logger.error(f"Lỗi khi lấy dữ liệu BTC: {e}")
                # Vẫn có thể phân tích được ETH mà không cần BTC
        
        # Phân tích dữ liệu
        self._analyze_market_trend()
        self._analyze_market_volatility()
        self._analyze_market_correlation()
        self._generate_trading_recommendation()
        
        # Cập nhật thời gian
        self.last_update = now
        
        return self.market_state
    
    def _analyze_market_trend(self):
        """
        Phân tích xu hướng thị trường dựa trên các đường MA và xu hướng giá.
        """
        if self.eth_data is None or len(self.eth_data) < 50:
            self.market_state['trend'] = 'UNKNOWN'
            self.market_state['conditions'].append('Không đủ dữ liệu để phân tích xu hướng')
            return
        
        # Tính các đường MA
        df = self.eth_data.copy()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        # Tính % thay đổi trong 7 ngày và 30 ngày
        df['change_7d'] = df['close'].pct_change(periods=7*24) * 100  # 7 ngày (với dữ liệu 1h)
        df['change_30d'] = df['close'].pct_change(periods=30*24) * 100  # 30 ngày
        
        # Lấy dữ liệu gần nhất
        latest = df.iloc[-1]
        prev_day = df.iloc[-24] if len(df) > 24 else df.iloc[0]
        
        # Tính xu hướng dựa trên vị trí của giá so với đường MA
        price_vs_ma20 = latest['close'] > latest['ma20']
        price_vs_ma50 = latest['close'] > latest['ma50']
        price_vs_ma200 = latest['close'] > latest['ma200']
        
        ma20_vs_ma50 = latest['ma20'] > latest['ma50']
        ma50_vs_ma200 = latest['ma50'] > latest['ma200']
        
        # Xu hướng dựa trên đường MA
        if price_vs_ma20 and price_vs_ma50 and price_vs_ma200 and ma20_vs_ma50 and ma50_vs_ma200:
            trend = 'STRONG_BULLISH'
        elif price_vs_ma20 and price_vs_ma50 and ma20_vs_ma50:
            trend = 'BULLISH'
        elif not price_vs_ma20 and not price_vs_ma50 and not price_vs_ma200 and not ma20_vs_ma50 and not ma50_vs_ma200:
            trend = 'STRONG_BEARISH'
        elif not price_vs_ma20 and not price_vs_ma50 and not ma20_vs_ma50:
            trend = 'BEARISH'
        else:
            # Kiểm tra xu hướng ngang
            price_range = (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()) / df['low'].rolling(window=14).min()
            is_sideways = price_range.iloc[-1] < 0.05  # Nếu biên độ dao động < 5% trong 14 nến 1h
            
            if is_sideways:
                trend = 'SIDEWAYS'
            else:
                trend = 'MIXED'
        
        # Đánh giá sức mạnh của xu hướng (mô hình ADX đơn giản)
        if len(df) >= 14:
            # Tính TR (True Range)
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            # Tính ATR (Average True Range)
            df['atr14'] = df['tr'].rolling(window=14).mean()
            
            # Tính +DM và -DM
            df['plus_dm'] = np.where(
                (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                np.maximum(df['high'] - df['high'].shift(1), 0),
                0
            )
            df['minus_dm'] = np.where(
                (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                np.maximum(df['low'].shift(1) - df['low'], 0),
                0
            )
            
            # Tính +DI14 và -DI14
            df['plus_di14'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr14'])
            df['minus_di14'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr14'])
            
            # Tính DX
            df['dx'] = 100 * abs(df['plus_di14'] - df['minus_di14']) / (df['plus_di14'] + df['minus_di14'])
            
            # Tính ADX
            df['adx'] = df['dx'].rolling(window=14).mean()
            
            # Đánh giá xu hướng dựa trên ADX
            adx_value = df['adx'].iloc[-1] if not pd.isna(df['adx'].iloc[-1]) else 0
            
            # Ghi nhận xu hướng và ADX
            self.market_state['trend'] = trend
            self.market_state['trend_strength'] = {
                'adx': round(adx_value, 2),
                'change_7d': round(latest['change_7d'], 2) if not pd.isna(latest['change_7d']) else 0,
                'change_30d': round(latest['change_30d'], 2) if not pd.isna(latest['change_30d']) else 0
            }
            
            # Thêm điều kiện về sức mạnh xu hướng
            if adx_value > 25:
                self.market_state['conditions'].append(f'Xu hướng {trend} mạnh (ADX = {adx_value:.1f})')
            else:
                self.market_state['conditions'].append(f'Xu hướng {trend} yếu (ADX = {adx_value:.1f})')
        else:
            self.market_state['trend'] = trend
            self.market_state['conditions'].append(f'Xu hướng {trend} (không đủ dữ liệu để tính ADX)')
    
    def _analyze_market_volatility(self):
        """
        Phân tích biến động của thị trường dựa trên ATR và Bollinger Bands.
        """
        if self.eth_data is None or len(self.eth_data) < 20:
            self.market_state['volatility'] = 'UNKNOWN'
            return
        
        # Tính ATR cho timeframe 1h
        df = self.eth_data.copy()
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr14'] = df['tr'].rolling(window=14).mean()
        
        # Tính % ATR so với giá
        df['atr_percent'] = df['atr14'] / df['close'] * 100
        
        # Tính Bollinger Bands
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['stddev'] = df['close'].rolling(window=20).std()
        df['bb_width'] = df['stddev'] / df['ma20'] * 100  # BB width as percentage
        
        # Lấy giá trị gần nhất
        latest_atr_pct = df['atr_percent'].iloc[-1] if not pd.isna(df['atr_percent'].iloc[-1]) else 0
        latest_bb_width = df['bb_width'].iloc[-1] if not pd.isna(df['bb_width'].iloc[-1]) else 0
        
        # So sánh với trung bình lịch sử để xác định mức độ biến động
        avg_atr_pct = df['atr_percent'].rolling(window=30).mean().iloc[-1] if len(df) >= 30 else df['atr_percent'].mean()
        avg_bb_width = df['bb_width'].rolling(window=30).mean().iloc[-1] if len(df) >= 30 else df['bb_width'].mean()
        
        # Phân loại biến động
        if latest_atr_pct > avg_atr_pct * 1.5 or latest_bb_width > avg_bb_width * 1.5:
            volatility = 'HIGH'
        elif latest_atr_pct < avg_atr_pct * 0.7 or latest_bb_width < avg_bb_width * 0.7:
            volatility = 'LOW'
        else:
            volatility = 'NORMAL'
        
        # Cập nhật trạng thái
        self.market_state['volatility'] = volatility
        self.market_state['volatility_metrics'] = {
            'atr_percent': round(latest_atr_pct, 2),
            'bb_width': round(latest_bb_width, 2),
            'atr_vs_avg': round(latest_atr_pct / avg_atr_pct, 2)
        }
        
        # Thêm điều kiện về biến động
        self.market_state['conditions'].append(f'Biến động thị trường: {volatility} (ATR = {latest_atr_pct:.2f}%, BB Width = {latest_bb_width:.2f}%)')
    
    def _analyze_market_correlation(self):
        """
        Phân tích tương quan giữa ETH và BTC
        """
        if self.eth_data is None or self.btc_data is None or len(self.eth_data) < 24 or len(self.btc_data) < 24:
            self.market_state['correlation'] = 'UNKNOWN'
            return
        
        # Chuẩn bị dữ liệu
        eth_df = self.eth_data.copy()
        btc_df = self.btc_data.copy()
        
        # Đảm bảo cùng index (timestamp)
        eth_df.set_index('open_time', inplace=True)
        btc_df.set_index('open_time', inplace=True)
        
        # Lấy dữ liệu chung
        common_index = eth_df.index.intersection(btc_df.index)
        eth_prices = eth_df.loc[common_index, 'close']
        btc_prices = btc_df.loc[common_index, 'close']
        
        # Tính tương quan
        if len(common_index) >= 24:  # Ít nhất 1 ngày dữ liệu
            # Tính % thay đổi giá
            eth_returns = eth_prices.pct_change().dropna()
            btc_returns = btc_prices.pct_change().dropna()
            
            # Kiểm tra lại dữ liệu
            if len(eth_returns) >= 24 and len(btc_returns) >= 24:
                # Tính hệ số tương quan Pearson
                corr = eth_returns.corr(btc_returns)
                
                # Phân loại tương quan
                if corr > 0.8:
                    correlation = 'VERY_HIGH'
                elif corr > 0.5:
                    correlation = 'HIGH'
                elif corr > 0.3:
                    correlation = 'MODERATE'
                elif corr > 0:
                    correlation = 'LOW'
                else:
                    correlation = 'NEGATIVE'
                
                # Tính beta (độ nhạy cảm của ETH so với BTC)
                if btc_returns.var() != 0:
                    cov = eth_returns.cov(btc_returns)
                    var = btc_returns.var()
                    beta = cov / var
                else:
                    beta = 0
                
                # Cập nhật trạng thái
                self.market_state['correlation'] = correlation
                self.market_state['correlation_metrics'] = {
                    'correlation': round(corr, 2),
                    'beta': round(beta, 2)
                }
                
                # Thêm điều kiện về tương quan
                self.market_state['conditions'].append(f'Tương quan với BTC: {correlation} (r = {corr:.2f}, beta = {beta:.2f})')
            else:
                self.market_state['correlation'] = 'UNKNOWN'
                self.market_state['conditions'].append('Không đủ dữ liệu để tính tương quan')
        else:
            self.market_state['correlation'] = 'UNKNOWN'
    
    def _generate_trading_recommendation(self):
        """
        Tạo khuyến nghị giao dịch dựa trên các điều kiện thị trường
        """
        trend = self.market_state.get('trend', 'UNKNOWN')
        volatility = self.market_state.get('volatility', 'UNKNOWN')
        correlation = self.market_state.get('correlation', 'UNKNOWN')
        
        # Mặc định là WAIT (chờ)
        recommend = 'WAIT'
        reasons = []
        
        # Phân tích xu hướng
        if trend in ['STRONG_BULLISH', 'BULLISH']:
            recommend = 'BUY_DIP'  # Mua khi điều chỉnh
            reasons.append(f'Xu hướng tăng ({trend})')
        elif trend in ['STRONG_BEARISH', 'BEARISH']:
            recommend = 'SELL_RALLY'  # Bán khi hồi phục
            reasons.append(f'Xu hướng giảm ({trend})')
        elif trend == 'SIDEWAYS':
            recommend = 'RANGE_TRADE'  # Giao dịch theo vùng
            reasons.append('Thị trường đi ngang')
        
        # Phân tích biến động
        if volatility == 'HIGH':
            # Nếu biến động cao, cần thận trọng
            if recommend == 'BUY_DIP':
                recommend = 'CAREFUL_BUY'
            elif recommend == 'SELL_RALLY':
                recommend = 'CAREFUL_SELL'
            reasons.append('Biến động cao, cần thận trọng')
        elif volatility == 'LOW':
            # Biến động thấp, có thể xem xét chiến lược breakout
            if recommend == 'RANGE_TRADE':
                recommend = 'PREPARE_BREAKOUT'
            reasons.append('Biến động thấp, có thể chuẩn bị breakout')
        
        # Phân tích tương quan
        if correlation == 'VERY_HIGH' and 'correlation_metrics' in self.market_state:
            beta = self.market_state['correlation_metrics'].get('beta', 0)
            if beta > 1.2:
                reasons.append(f'ETH biến động mạnh hơn BTC (beta = {beta:.2f})')
            elif beta < 0.8:
                reasons.append(f'ETH biến động yếu hơn BTC (beta = {beta:.2f})')
        
        # Xem xét các điều kiện đặc biệt
        trend_strength = self.market_state.get('trend_strength', {})
        adx = trend_strength.get('adx', 0)
        change_7d = trend_strength.get('change_7d', 0)
        
        # Quá mua/quá bán
        if change_7d > 20 and trend in ['STRONG_BULLISH', 'BULLISH']:
            recommend = 'CAREFUL_BUY'  # Thận trọng mua vì có thể quá mua
            reasons.append(f'Tăng mạnh trong 7 ngày ({change_7d:.1f}%), cẩn thận quá mua')
        elif change_7d < -20 and trend in ['STRONG_BEARISH', 'BEARISH']:
            recommend = 'CAREFUL_SELL'  # Thận trọng bán vì có thể quá bán
            reasons.append(f'Giảm mạnh trong 7 ngày ({change_7d:.1f}%), cẩn thận quá bán')
        
        # Xu hướng mạnh
        if adx > 30:
            if trend in ['STRONG_BULLISH', 'BULLISH']:
                recommend = 'STRONG_BUY'
                reasons.append(f'Xu hướng tăng mạnh (ADX = {adx:.1f})')
            elif trend in ['STRONG_BEARISH', 'BEARISH']:
                recommend = 'STRONG_SELL'
                reasons.append(f'Xu hướng giảm mạnh (ADX = {adx:.1f})')
        
        # Cập nhật trạng thái
        self.market_state['recommend'] = recommend
        self.market_state['reasons'] = reasons
        
        # Thêm lời khuyên cụ thể
        if recommend == 'BUY_DIP':
            self.market_state['advice'] = 'Mua khi giá điều chỉnh về vùng hỗ trợ'
        elif recommend == 'SELL_RALLY':
            self.market_state['advice'] = 'Bán khi giá hồi phục lên vùng kháng cự'
        elif recommend == 'RANGE_TRADE':
            self.market_state['advice'] = 'Giao dịch trong biên độ, mua tại hỗ trợ và bán tại kháng cự'
        elif recommend == 'CAREFUL_BUY':
            self.market_state['advice'] = 'Thận trọng khi mua, giảm kích thước vị thế'
        elif recommend == 'CAREFUL_SELL':
            self.market_state['advice'] = 'Thận trọng khi bán, giảm kích thước vị thế'
        elif recommend == 'PREPARE_BREAKOUT':
            self.market_state['advice'] = 'Chuẩn bị giao dịch breakout, đặt lệnh chờ khi giá vượt kháng cự hoặc phá vỡ hỗ trợ'
        elif recommend == 'STRONG_BUY':
            self.market_state['advice'] = 'Mua và nắm giữ theo xu hướng tăng mạnh'
        elif recommend == 'STRONG_SELL':
            self.market_state['advice'] = 'Bán và nắm giữ vị thế short theo xu hướng giảm mạnh'
        else:
            self.market_state['advice'] = 'Chờ đợi tín hiệu rõ ràng hơn từ thị trường'

class DecisionSupport:
    """
    Hỗ trợ quyết định giao dịch với các mức TP, SL và quản lý rủi ro
    """
    def __init__(self, market_filter=None):
        """
        Khởi tạo hệ thống hỗ trợ quyết định
        
        Args:
            market_filter (MarketFilter): Bộ lọc thị trường
        """
        self.market_filter = market_filter
    
    def analyze_trade_setup(self, prediction, data):
        """
        Phân tích thiết lập giao dịch từ dự đoán và dữ liệu
        
        Args:
            prediction (dict): Dự đoán từ prediction engine
            data (pd.DataFrame): Dữ liệu giá gần đây
            
        Returns:
            dict: Phân tích thiết lập giao dịch với các mức TP/SL
        """
        if data is None or len(data) < 20:
            return {"error": "Không đủ dữ liệu để phân tích"}
        
        try:
            # Lấy dữ liệu thị trường tổng thể nếu có
            market_state = {}
            if self.market_filter:
                market_state = self.market_filter.update(eth_data=data)
            
            # Lấy giá hiện tại và xu hướng dự đoán
            current_price = data['close'].iloc[-1]
            predicted_trend = prediction.get('trend', 'NEUTRAL')
            
            # Tính ATR để đặt SL/TP hợp lý
            data['tr'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )
            data['atr14'] = data['tr'].rolling(window=14).mean()
            atr = data['atr14'].iloc[-1]
            
            # Lấy các mức hỗ trợ/kháng cự
            support_levels = prediction.get('support_levels', [])
            resistance_levels = prediction.get('resistance_levels', [])
            
            # Chuẩn bị kết quả
            result = {
                "current_price": current_price,
                "predicted_trend": predicted_trend,
                "atr": atr,
                "risk_reward_ratio": prediction.get('risk_reward_ratio', 0),
                "market_state": market_state
            }
            
            # Tính các mức TP, SL dựa trên xu hướng
            if predicted_trend == 'LONG':
                # Tìm mức kháng cự gần nhất làm TP
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > current_price], 
                                        default=current_price * 1.01)
                    result['tp1'] = nearest_resistance
                else:
                    # Nếu không có mức kháng cự, sử dụng ATR
                    result['tp1'] = current_price + 1.5 * atr
                
                # Tìm mức hỗ trợ gần nhất làm SL
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < current_price], 
                                      default=current_price * 0.99)
                    result['sl'] = nearest_support
                else:
                    # Nếu không có mức hỗ trợ, sử dụng ATR
                    result['sl'] = current_price - 1 * atr
                
                # Thêm TP2 và TP3 xa hơn
                result['tp2'] = current_price + 2.5 * atr
                result['tp3'] = current_price + 4 * atr
                
            elif predicted_trend == 'SHORT':
                # Tìm mức hỗ trợ gần nhất làm TP
                if support_levels:
                    nearest_support = max([s for s in support_levels if s < current_price], 
                                      default=current_price * 0.99)
                    result['tp1'] = nearest_support
                else:
                    # Nếu không có mức hỗ trợ, sử dụng ATR
                    result['tp1'] = current_price - 1.5 * atr
                
                # Tìm mức kháng cự gần nhất làm SL
                if resistance_levels:
                    nearest_resistance = min([r for r in resistance_levels if r > current_price], 
                                        default=current_price * 1.01)
                    result['sl'] = nearest_resistance
                else:
                    # Nếu không có mức kháng cự, sử dụng ATR
                    result['sl'] = current_price + 1 * atr
                
                # Thêm TP2 và TP3 xa hơn
                result['tp2'] = current_price - 2.5 * atr
                result['tp3'] = current_price - 4 * atr
            
            # Tính lại tỷ lệ risk/reward
            if 'sl' in result and 'tp1' in result:
                risk = abs(current_price - result['sl'])
                reward = abs(current_price - result['tp1'])
                if risk > 0:
                    result['risk_reward_ratio'] = round(reward / risk, 2)
            
            # Tính kích thước vị thế đề xuất
            max_risk_percent = 2.0  # Tối đa 2% vốn cho mỗi giao dịch
            # Điều chỉnh theo mức độ tin cậy của dự đoán
            confidence = prediction.get('confidence', 0.5)
            if confidence < 0.7:
                max_risk_percent *= 0.5  # Giảm một nửa nếu độ tin cậy thấp
            
            # Tính position size theo % rủi ro
            if 'sl' in result:
                price_diff_percent = abs(current_price - result['sl']) / current_price * 100
                if price_diff_percent > 0:
                    result['position_size_percent'] = round(max_risk_percent / price_diff_percent * 100, 2)
                    # Giới hạn tối đa 30% vốn cho mỗi giao dịch
                    result['position_size_percent'] = min(30, result['position_size_percent'])
            
            # Thêm điều kiện thị trường và đề xuất
            if market_state:
                market_recommend = market_state.get('recommend', 'WAIT')
                market_trend = market_state.get('trend', 'UNKNOWN')
                
                # Xác định nếu dự đoán phù hợp với điều kiện thị trường
                trend_alignment = False
                if (predicted_trend == 'LONG' and market_trend in ['STRONG_BULLISH', 'BULLISH']) or \
                   (predicted_trend == 'SHORT' and market_trend in ['STRONG_BEARISH', 'BEARISH']):
                    trend_alignment = True
                    result['market_alignment'] = 'HIGH'
                elif (predicted_trend == 'LONG' and market_trend == 'MIXED') or \
                     (predicted_trend == 'SHORT' and market_trend == 'MIXED'):
                    result['market_alignment'] = 'MODERATE'
                elif (predicted_trend == 'LONG' and market_trend in ['STRONG_BEARISH', 'BEARISH']) or \
                     (predicted_trend == 'SHORT' and market_trend in ['STRONG_BULLISH', 'BULLISH']):
                    result['market_alignment'] = 'LOW'
                    # Nếu đi ngược xu hướng, giảm position size
                    if 'position_size_percent' in result:
                        result['position_size_percent'] *= 0.5
                else:
                    result['market_alignment'] = 'NEUTRAL'
                
                # Thêm lời khuyên
                if trend_alignment:
                    result['recommendation'] = f"Dự đoán phù hợp với xu hướng thị trường. {market_state.get('advice', '')}"
                else:
                    result['recommendation'] = f"Dự đoán không phù hợp với xu hướng thị trường. Cân nhắc thận trọng."
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích thiết lập giao dịch: {e}")
            return {"error": f"Lỗi phân tích: {str(e)}"}