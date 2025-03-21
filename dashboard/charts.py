"""
Chart creation functions for the Streamlit dashboard.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import config
from utils.feature_engineering import TechnicalIndicators

def plot_candlestick_chart(df):
    """
    Create a candlestick chart with volume.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        go.Figure: Plotly figure with candlestick chart
    """
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3],
        subplot_titles=('ETHUSDT', 'Volume')
    )
    
    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='ETHUSDT'
        ),
        row=1, col=1
    )
    
    # Add volume trace
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Add some moving averages
    for period in [9, 21]:
        ma = df['close'].rolling(window=period).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ma,
                mode='lines',
                line=dict(width=1),
                name=f'MA {period}'
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='ETHUSDT Price Chart',
        xaxis_title='Date',
        yaxis_title='Price (USDT)',
        height=600,
        margin=dict(l=50, r=50, b=50, t=70),
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    # Style candlesticks
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=3, label="3h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=24, label="1d", step="hour", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
    )
    
    return fig

def plot_technical_indicators(df):
    """
    Create a chart with technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        go.Figure: Plotly figure with technical indicators
    """
    # Create figure with multiple subplots
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=('Bollinger Bands', 'RSI', 'MACD')
    )
    
    # Initialize technical indicators
    ti = TechnicalIndicators()
    
    # Calculate Bollinger Bands
    upper, middle, lower = ti.BBANDS(
        df['close'],
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2
    )
    
    # Plot price and Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            name='Price',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=upper,
            name='Upper BB',
            line=dict(color='blue', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=middle,
            name='Middle BB',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=lower,
            name='Lower BB',
            line=dict(color='blue', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ),
        row=1, col=1
    )
    
    # Calculate and plot RSI
    rsi = ti.RSI(df['close'], timeperiod=14)
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rsi,
            name='RSI',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    # Add RSI levels
    fig.add_shape(
        type='line',
        x0=df.index[0],
        x1=df.index[-1],
        y0=70,
        y1=70,
        line=dict(color='red', width=1, dash='dash'),
        row=2, col=1
    )
    
    fig.add_shape(
        type='line',
        x0=df.index[0],
        x1=df.index[-1],
        y0=30,
        y1=30,
        line=dict(color='green', width=1, dash='dash'),
        row=2, col=1
    )
    
    # Calculate and plot MACD
    macd, signal, hist = ti.MACD(
        df['close'],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=macd,
            name='MACD',
            line=dict(color='blue', width=1)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=signal,
            name='Signal',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    # Add MACD histogram
    colors = ['green' if val >= 0 else 'red' for val in hist]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=hist,
            marker_color=colors,
            name='Histogram'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        margin=dict(l=50, r=50, b=50, t=70),
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def plot_prediction_history(predictions):
    """
    Create a chart with prediction history.
    
    Args:
        predictions (list): List of prediction dictionaries
        
    Returns:
        go.Figure: Plotly figure with prediction history
    """
    # Khởi tạo biến fig ở cấp độ hàm để tránh lỗi unbound variable
    fig = go.Figure()
    
    try:
        # Đảm bảo predictions không rỗng
        if not predictions or len(predictions) == 0:
            # Trả về biểu đồ trống nếu không có dữ liệu dự đoán
            fig.update_layout(
                title='Prediction History',
                xaxis_title='Time',
                yaxis_title='Confidence',
                height=400,
                annotations=[dict(
                    text='No prediction data available',
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )]
            )
            return fig
            
        # Tạo bản sao của dữ liệu để tránh thay đổi
        pred_copy = predictions.copy()
        
        # Chuẩn hóa dữ liệu trước khi chuyển thành DataFrame
        for p in pred_copy:
            # Chuẩn hóa trường trend để đảm bảo chữ thường
            if 'trend' in p and p['trend'] is not None:
                p['trend'] = str(p['trend']).lower()
                
            # Đảm bảo confidence là số
            if 'confidence' in p:
                try:
                    p['confidence'] = float(p['confidence'])
                except (ValueError, TypeError):
                    p['confidence'] = 0.0
        
        # Chuyển predictions thành DataFrame
        df = pd.DataFrame(pred_copy)
        
        # Kiểm tra xem columns cần thiết có tồn tại không
        if 'timestamp' not in df.columns or 'confidence' not in df.columns or 'trend' not in df.columns:
            raise ValueError("Prediction data missing required columns: timestamp, confidence, or trend")
        
        # Chuyển đổi timestamp thành datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sắp xếp theo timestamp
        df = df.sort_values('timestamp')
        
        # Xác định bản đồ màu cho trend
        trend_color_map = {
            'long': 'green',
            'short': 'red',
            'neutral': 'gray'
        }
        
        # Tạo figure mới
        fig = go.Figure()
        
        # Thêm đường confidence
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['confidence'],
                mode='lines+markers',
                name='Confidence',
                line=dict(color='blue', width=2),
                marker=dict(
                    size=10,
                    color=[trend_color_map.get(str(trend).lower(), 'gray') for trend in df['trend']]
                )
            )
        )
        
        # Thêm ngưỡng confidence
        fig.add_shape(
            type='line',
            x0=df['timestamp'].min(),
            x1=df['timestamp'].max(),
            y0=config.CONFIDENCE_THRESHOLD,
            y1=config.CONFIDENCE_THRESHOLD,
            line=dict(color='orange', width=1, dash='dash')
        )
        
        # Thêm chỉ báo trend dưới dạng annotations
        for i, row in df.iterrows():
            try:
                trend_text = str(row['trend']).upper()
                trend_color = trend_color_map.get(str(row['trend']).lower(), 'gray')
                
                fig.add_annotation(
                    x=row['timestamp'],
                    y=row['confidence'],
                    text=trend_text,
                    showarrow=False,
                    yshift=15,
                    font=dict(
                        size=10,
                        color=trend_color
                    )
                )
            except Exception as e:
                print(f"Skipping annotation for row {i} due to error: {str(e)}")
                continue
        
        # Update layout trong try block
        fig.update_layout(
            title='Prediction History',
            xaxis_title='Time',
            yaxis_title='Confidence',
            height=400,
            margin=dict(l=50, r=50, b=50, t=70),
            template='plotly_white',
            yaxis=dict(range=[0, 1])
        )
        
    except Exception as e:
        print(f"Error plotting prediction history: {str(e)}")
        # Cấu hình biểu đồ trống nếu có lỗi
        fig.update_layout(
            title='Prediction History',
            xaxis_title='Time',
            yaxis_title='Confidence',
            height=400,
            annotations=[dict(
                text=f'Error generating chart: {str(e)}',
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )]
        )
    
    # Trả về biểu đồ kết quả
    return fig

def plot_confidence_distribution(predictions):
    """
    Create a chart with confidence distribution by trend.
    
    Args:
        predictions (list): List of prediction dictionaries
        
    Returns:
        go.Figure: Plotly figure with confidence distribution
    """
    # Khởi tạo biến fig ở cấp độ hàm
    fig = go.Figure()
    
    try:
        # Kiểm tra nếu predictions rỗng
        if not predictions or len(predictions) == 0:
            fig.update_layout(
                title='Confidence Distribution by Trend',
                xaxis_title='Confidence',
                yaxis_title='Count',
                height=300,
                annotations=[dict(
                    text='No prediction data available',
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )]
            )
            return fig
            
        # Tạo bản sao của dữ liệu để tránh thay đổi
        pred_copy = predictions.copy()
        
        # Chuẩn hóa dữ liệu trước khi chuyển thành DataFrame
        for p in pred_copy:
            # Chuẩn hóa trường trend để đảm bảo chữ thường
            if 'trend' in p and p['trend'] is not None:
                p['trend'] = str(p['trend']).lower()
                
            # Đảm bảo confidence là số
            if 'confidence' in p:
                try:
                    p['confidence'] = float(p['confidence'])
                except (ValueError, TypeError):
                    p['confidence'] = 0.0
        
        # Chuyển predictions thành DataFrame
        df = pd.DataFrame(pred_copy)
        
        # Kiểm tra xem columns cần thiết có tồn tại không
        if 'confidence' not in df.columns or 'trend' not in df.columns:
            raise ValueError("Prediction data missing required columns: confidence, or trend")
        
        # Xác định bản đồ màu cho trend
        trend_color_map = {
            'long': 'green',
            'short': 'red',
            'neutral': 'gray'
        }
        
        # Group by trend
        trends = df['trend'].unique()
        
        # Create histogram for each trend
        for trend in trends:
            trend_data = df[df['trend'] == trend]
            
            fig.add_trace(
                go.Histogram(
                    x=trend_data['confidence'],
                    name=str(trend).upper(),
                    marker_color=trend_color_map.get(str(trend).lower(), 'blue'),
                    opacity=0.7,
                    nbinsx=10
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Confidence Distribution by Trend',
            xaxis_title='Confidence',
            yaxis_title='Count',
            height=300,
            margin=dict(l=50, r=50, b=50, t=70),
            template='plotly_white',
            barmode='overlay'
        )
        
    except Exception as e:
        print(f"Error plotting confidence distribution: {str(e)}")
        # Trả về biểu đồ trống nếu có lỗi
        fig.update_layout(
            title='Confidence Distribution by Trend',
            xaxis_title='Confidence',
            yaxis_title='Count',
            height=300,
            annotations=[dict(
                text=f'Error generating chart: {str(e)}',
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )]
        )
    
    return fig

def plot_model_accuracy(evaluation_results):
    """
    Create a chart comparing model accuracies.
    
    Args:
        evaluation_results (dict): Dictionary with model evaluation results
        
    Returns:
        go.Figure: Plotly figure with model accuracy comparison
    """
    # Khởi tạo biến fig ở cấp độ hàm
    fig = go.Figure()
    
    try:
        # Kiểm tra nếu evaluation_results rỗng
        if not evaluation_results or not isinstance(evaluation_results, dict):
            fig.update_layout(
                title='Model Accuracy Comparison',
                xaxis_title='Model',
                yaxis_title='Accuracy',
                height=400,
                annotations=[dict(
                    text='No model evaluation data available',
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )]
            )
            return fig
            
        # Extract model names and accuracies
        models = []
        accuracies = []
        
        for model, results in evaluation_results.items():
            # Đảm bảo results là dict
            if not isinstance(results, dict):
                continue
                
            models.append(str(model))
            # Get accuracy, falling back to 0 if not available
            try:
                accuracy = float(results.get('accuracy', 0))
            except (ValueError, TypeError):
                accuracy = 0.0
            accuracies.append(accuracy)
        
        # Kiểm tra nếu không có dữ liệu hợp lệ
        if not models or not accuracies:
            fig.update_layout(
                title='Model Accuracy Comparison',
                xaxis_title='Model',
                yaxis_title='Accuracy',
                height=400,
                annotations=[dict(
                    text='No valid model accuracy data available',
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )]
            )
            return fig
        
        # Create color map
        colors = {
            'lstm': 'rgb(55, 83, 109)',
            'transformer': 'rgb(26, 118, 255)',
            'cnn': 'rgb(86, 175, 211)',
            'historical': 'rgb(158, 202, 225)',
            'meta': 'rgb(0, 128, 0)'
        }
        
        # Use default colors for unknown models
        bar_colors = [colors.get(model.lower(), 'lightgray') for model in models]
        
        # Thêm bar chart
        fig.add_trace(
            go.Bar(
                x=models,
                y=accuracies,
                marker_color=bar_colors
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Model Accuracy Comparison',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            height=400,
            margin=dict(l=50, r=50, b=50, t=70),
            template='plotly_white',
            yaxis=dict(range=[0, 1])
        )
        
    except Exception as e:
        print(f"Error plotting model accuracy: {str(e)}")
        # Trả về biểu đồ trống nếu có lỗi
        fig.update_layout(
            title='Model Accuracy Comparison',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            height=400,
            annotations=[dict(
                text=f'Error generating chart: {str(e)}',
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )]
        )
    
    return fig
