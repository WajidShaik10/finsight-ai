# quant_rag_agent/modules/agent.py

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from groq import Groq
from serpapi import GoogleSearch
from quant_rag_agent.modules.retriever import DocumentRetriever

class QuantAgent:
    def __init__(self, collection_name="quant_documents"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY", "").strip())
        self.retriever = DocumentRetriever(collection_name=collection_name)
        self.model = "llama-3.3-70b-versatile"
        self.history = []
        self.serpapi_key = os.environ.get("SERPAPI_KEY", "").strip()
        print("Agent ready!")

    # ─────────────────────────────────────────
    # BASIC TOOLS
    # ─────────────────────────────────────────

    def search_documents(self, query):
        print(f"\n📄 Searching documents: '{query}'")
        chunks = self.retriever.retrieve(query, top_k=5)
        return "\n\n".join(chunks)

    def search_web(self, query):
        print(f"\n🌐 Searching web: '{query}'")
        try:
            search = GoogleSearch({
                "q": query,
                "api_key": self.serpapi_key,
                "num": 5
            })
            results = search.get_dict()
            snippets = []
            if "answer_box" in results:
                box = results["answer_box"]
                if "answer" in box:
                    snippets.append(f"Direct answer: {box['answer']}")
                elif "snippet" in box:
                    snippets.append(f"Summary: {box['snippet']}")
            if "organic_results" in results:
                for r in results["organic_results"][:4]:
                    if "snippet" in r:
                        snippets.append(f"{r['title']}: {r['snippet']}")
            return "\n\n".join(snippets) if snippets else "No results found"
        except Exception as e:
            return f"Web search error: {e}"

    # ─────────────────────────────────────────
    # MARKET DATA TOOLS
    # ─────────────────────────────────────────

    def get_stock_data(self, ticker):
        print(f"\n📈 Getting stock data: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
            prev_close = info.get('previousClose', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            pe_ratio = info.get('trailingPE', 'N/A')
            eps = info.get('trailingEps', 'N/A')
            revenue = info.get('totalRevenue', 'N/A')
            profit_margin = info.get('profitMargins', 'N/A')
            week_high = info.get('fiftyTwoWeekHigh', 'N/A')
            week_low = info.get('fiftyTwoWeekLow', 'N/A')
            dividend = info.get('dividendYield', 'N/A')
            beta = info.get('beta', 'N/A')
            name = info.get('longName', ticker)

            if isinstance(market_cap, (int, float)):
                market_cap = f"${market_cap/1e9:.2f}B"
            if isinstance(revenue, (int, float)):
                revenue = f"${revenue/1e9:.2f}B"
            if isinstance(profit_margin, float):
                profit_margin = f"{profit_margin*100:.2f}%"
            if isinstance(dividend, float):
                dividend = f"{dividend*100:.2f}%"
            if isinstance(price, (int, float)) and isinstance(prev_close, (int, float)):
                change = price - prev_close
                change_pct = (change / prev_close) * 100
                change_str = f"+${change:.2f} (+{change_pct:.2f}%)" if change >= 0 else f"-${abs(change):.2f} ({change_pct:.2f}%)"
            else:
                change_str = "N/A"

            return f"""**{name} ({ticker.upper()})**

**Current Price:** ${price}
**Change:** {change_str}
**Previous Close:** ${prev_close}

**Key Metrics:**
- Market Cap: {market_cap}
- P/E Ratio: {pe_ratio}
- EPS: ${eps}
- Revenue: {revenue}
- Profit Margin: {profit_margin}
- Beta: {beta}
- Dividend Yield: {dividend}

**52-Week Range:** ${week_low} — ${week_high}"""
        except Exception as e:
            print(f"yfinance failed for {ticker}: {e}, falling back to web search")
            return self._llm_answer(
                f"What is the current stock price and key metrics for {ticker}?",
                self.search_web(f"{ticker} stock price today market cap PE ratio"),
                "stock market analyst"
            )

    def get_historical_data(self, ticker, period="1y"):
        print(f"\n📊 Getting historical data: {ticker} ({period})")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                return self.search_web(f"{ticker} stock price history {period}")
            prices = hist['Close'].tolist()
            dates = [str(d.date()) for d in hist.index.tolist()]
            step = max(1, len(dates) // 12)
            sampled_dates = dates[::step]
            sampled_prices = prices[::step]
            result = f"**{ticker.upper()} Historical Prices ({period}):**\n\n"
            for date, price in zip(sampled_dates, sampled_prices):
                result += f"- {date}: ${price:.2f}\n"
            start_price = prices[0]
            end_price = prices[-1]
            total_return = ((end_price - start_price) / start_price) * 100
            result += f"\n**Total Return ({period}):** {total_return:+.2f}%"
            return result
        except Exception as e:
            return self.search_web(f"{ticker} stock price history {period}")

    def compare_stocks(self, tickers):
        print(f"\n⚖️ Comparing stocks: {tickers}")
        try:
            results = []
            for ticker in tickers:
                stock = yf.Ticker(ticker.strip())
                info = stock.info
                name = info.get('longName', ticker)
                price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
                market_cap = info.get('marketCap', 0)
                pe = info.get('trailingPE', 'N/A')
                margin = info.get('profitMargins', 0)
                revenue = info.get('totalRevenue', 0)
                beta = info.get('beta', 'N/A')
                results.append({
                    'name': name,
                    'ticker': ticker.upper(),
                    'price': f"${price}" if isinstance(price, (int, float)) else price,
                    'market_cap': f"${market_cap/1e9:.2f}B" if isinstance(market_cap, (int, float)) else 'N/A',
                    'pe': f"{pe:.2f}" if isinstance(pe, (int, float)) else pe,
                    'margin': f"{margin*100:.2f}%" if isinstance(margin, float) else 'N/A',
                    'revenue': f"${revenue/1e9:.2f}B" if isinstance(revenue, (int, float)) else 'N/A',
                    'beta': f"{beta:.2f}" if isinstance(beta, (int, float)) else beta
                })
            output = "| Metric | " + " | ".join([r['ticker'] for r in results]) + " |\n"
            output += "|--------|" + "|---------|" * len(results) + "\n"
            output += "| Price | " + " | ".join([r['price'] for r in results]) + " |\n"
            output += "| Market Cap | " + " | ".join([r['market_cap'] for r in results]) + " |\n"
            output += "| P/E Ratio | " + " | ".join([r['pe'] for r in results]) + " |\n"
            output += "| Profit Margin | " + " | ".join([r['margin'] for r in results]) + " |\n"
            output += "| Revenue | " + " | ".join([r['revenue'] for r in results]) + " |\n"
            output += "| Beta | " + " | ".join([r['beta'] for r in results]) + " |\n"
            return output
        except Exception as e:
            return self.search_web(f"compare {' vs '.join(tickers)} stock price revenue profit")

    def get_financials(self, ticker):
        print(f"\n💰 Getting financials: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            def fmt(val):
                if isinstance(val, (int, float)):
                    return f"${val/1e9:.2f}B"
                return str(val)
            def fmt_pct(val):
                if isinstance(val, float):
                    return f"{val*100:.2f}%"
                return str(val)
            return f"""**{ticker.upper()} Financial Summary**

**Income Statement:**
- Revenue: {fmt(info.get('totalRevenue', 'N/A'))}
- Gross Profit: {fmt(info.get('grossProfits', 'N/A'))}
- Net Income: {fmt(info.get('netIncomeToCommon', 'N/A'))}
- EBITDA: {fmt(info.get('ebitda', 'N/A'))}

**Balance Sheet:**
- Total Debt: {fmt(info.get('totalDebt', 'N/A'))}
- Total Cash: {fmt(info.get('totalCash', 'N/A'))}
- Operating Cash Flow: {fmt(info.get('operatingCashflow', 'N/A'))}

**Profitability:**
- Return on Equity (ROE): {fmt_pct(info.get('returnOnEquity', 'N/A'))}
- Return on Assets (ROA): {fmt_pct(info.get('returnOnAssets', 'N/A'))}"""
        except Exception as e:
            return self._llm_answer(
                f"What are the key financial metrics for {ticker}?",
                self.search_web(f"{ticker} revenue net income profit margin financial results 2024"),
                "financial analyst"
            )

    # ─────────────────────────────────────────
    # QUANT TOOLS
    # ─────────────────────────────────────────

    def calculate_sharpe_ratio(self, ticker, period="1y", risk_free_rate=0.05):
        print(f"\n📐 Calculating Sharpe Ratio: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                return f"No data found for {ticker}"
            returns = hist['Close'].pct_change().dropna()
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.cummax()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            win_rate = (returns > 0).sum() / len(returns) * 100
            rating = "🔴 Poor" if sharpe < 0 else "🟡 Below Average" if sharpe < 0.5 else "🟢 Good" if sharpe < 1 else "⭐ Great" if sharpe < 2 else "🏆 Excellent"
            return f"""**Sharpe Ratio Analysis — {ticker.upper()} ({period})**

**Sharpe Ratio: {sharpe:.3f}** {rating}

**Performance Metrics:**
- Annual Return: {annual_return*100:.2f}%
- Annual Volatility: {annual_volatility*100:.2f}%
- Risk-Free Rate: {risk_free_rate*100:.2f}%
- Max Drawdown: {max_drawdown:.2f}%
- Win Rate: {win_rate:.1f}%

**Benchmark:**
- < 0: Losing money after risk adjustment
- 0-1: Below average
- 1-2: Good
- 2-3: Very good
- > 3: Excellent"""
        except Exception as e:
            return f"Error calculating Sharpe Ratio: {e}"

    def backtest_strategy(self, ticker, strategy="SMA", period="2y"):
        print(f"\n🔬 Backtesting {strategy} strategy on {ticker}")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                return f"No data found for {ticker}"
            prices = hist['Close'].copy()
            signals = pd.Series(0, index=prices.index)
            if strategy.upper() == "SMA":
                sma_short = prices.rolling(window=20).mean()
                sma_long = prices.rolling(window=50).mean()
                signals[sma_short > sma_long] = 1
                signals[sma_short < sma_long] = -1
                strategy_name = "SMA Crossover (20/50)"
            elif strategy.upper() == "RSI":
                delta = prices.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                signals[rsi < 30] = 1
                signals[rsi > 70] = -1
                strategy_name = "RSI (30/70)"
            elif strategy.upper() == "MOMENTUM":
                momentum = prices.pct_change(20)
                signals[momentum > 0.05] = 1
                signals[momentum < -0.05] = -1
                strategy_name = "Momentum (20-day)"
            else:
                return f"Unknown strategy: {strategy}. Try SMA, RSI, or MOMENTUM"
            daily_returns = prices.pct_change()
            strategy_returns = signals.shift(1) * daily_returns
            total_return = (1 + strategy_returns).prod() - 1
            buy_hold_return = (prices.iloc[-1] / prices.iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility != 0 else 0
            cumulative = (1 + strategy_returns).cumprod()
            rolling_max = cumulative.cummax()
            max_drawdown = ((cumulative - rolling_max) / rolling_max).min() * 100
            trades = signals.diff().abs().sum() / 2
            winner = "✅ Strategy BEATS buy & hold!" if total_return > buy_hold_return else "❌ Buy & hold BEATS strategy"
            return f"""**Backtest Results — {ticker.upper()} — {strategy_name}**
**Period:** {period}

**Strategy Performance:**
- Total Return: **{total_return*100:.2f}%**
- Buy & Hold Return: **{buy_hold_return*100:.2f}%**
- {winner}

**Risk Metrics:**
- Annual Return: {annual_return*100:.2f}%
- Annual Volatility: {volatility*100:.2f}%
- Sharpe Ratio: {sharpe:.3f}
- Max Drawdown: {max_drawdown:.2f}%

**Trade Statistics:**
- Total Trades: {int(trades)}
- Avg trades/year: {int(trades/(len(prices)/252))}/year

**Verdict:** {"This strategy outperformed passive investing." if total_return > buy_hold_return else "Simply holding the stock would have been more profitable."}"""
        except Exception as e:
            return f"Error backtesting: {e}"

    def optimize_portfolio(self, tickers, period="1y"):
        print(f"\n🎯 Optimizing portfolio: {tickers}")
        try:
            data = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker.strip())
                hist = stock.history(period=period)
                if not hist.empty:
                    data[ticker.strip().upper()] = hist['Close']
            if len(data) < 2:
                return "Need at least 2 valid tickers for portfolio optimization"
            prices_df = pd.DataFrame(data)
            returns_df = prices_df.pct_change().dropna()
            mean_returns = returns_df.mean() * 252
            cov_matrix = returns_df.cov() * 252
            n = len(data)
            def portfolio_metrics(weights):
                ret = np.dot(weights, mean_returns)
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = ret / vol
                return ret, vol, sharpe
            def neg_sharpe(weights):
                return -portfolio_metrics(weights)[2]
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(n))
            initial = np.array([1/n] * n)
            result = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_weights = result.x
            opt_return, opt_vol, opt_sharpe = portfolio_metrics(optimal_weights)
            eq_weights = np.array([1/n] * n)
            eq_return, eq_vol, eq_sharpe = portfolio_metrics(eq_weights)
            output = f"""**Portfolio Optimization Results**
**Tickers:** {', '.join(data.keys())}
**Period:** {period}

**Optimal Weights (Max Sharpe Ratio):**
"""
            for ticker, weight in zip(data.keys(), optimal_weights):
                bar = "█" * int(weight * 20)
                output += f"- **{ticker}:** {weight*100:.1f}% {bar}\n"
            output += f"""
**Optimal Portfolio Metrics:**
- Expected Annual Return: **{opt_return*100:.2f}%**
- Annual Volatility: **{opt_vol*100:.2f}%**
- Sharpe Ratio: **{opt_sharpe:.3f}**

**Equal Weight Portfolio (for comparison):**
- Expected Annual Return: {eq_return*100:.2f}%
- Annual Volatility: {eq_vol*100:.2f}%
- Sharpe Ratio: {eq_sharpe:.3f}

**Improvement:** Optimal portfolio has {((opt_sharpe/eq_sharpe)-1)*100:.1f}% better risk-adjusted return than equal weighting."""
            return output
        except Exception as e:
            return f"Error optimizing portfolio: {e}"

    def monte_carlo_simulation(self, ticker, days=30, simulations=1000):
        print(f"\n🎲 Running Monte Carlo simulation: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if hist.empty:
                return f"No data found for {ticker}"
            prices = hist['Close']
            current_price = prices.iloc[-1]
            daily_returns = prices.pct_change().dropna()
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            np.random.seed(42)
            simulation_results = []
            for _ in range(simulations):
                price_path = [current_price]
                for _ in range(days):
                    shock = np.random.normal(mean_return, std_return)
                    price_path.append(price_path[-1] * (1 + shock))
                simulation_results.append(price_path[-1])
            simulation_results = np.array(simulation_results)
            mean_price = np.mean(simulation_results)
            median_price = np.median(simulation_results)
            std_price = np.std(simulation_results)
            percentile_5 = np.percentile(simulation_results, 5)
            percentile_25 = np.percentile(simulation_results, 25)
            percentile_75 = np.percentile(simulation_results, 75)
            percentile_95 = np.percentile(simulation_results, 95)
            prob_profit = (simulation_results > current_price).mean() * 100
            prob_loss_10 = (simulation_results < current_price * 0.9).mean() * 100
            prob_gain_10 = (simulation_results > current_price * 1.1).mean() * 100
            expected_return = ((mean_price - current_price) / current_price) * 100
            return f"""**Monte Carlo Simulation — {ticker.upper()}**
**Simulations:** {simulations:,} | **Forecast Period:** {days} days

**Current Price:** ${current_price:.2f}

**Price Forecast ({days} days):**
- Most Likely Price: **${mean_price:.2f}**
- Median Price: **${median_price:.2f}**
- Expected Return: **{expected_return:+.2f}%**

**Price Range (90% Confidence):**
- 🔴 Worst Case (5th percentile): ${percentile_5:.2f}
- 🟡 Lower Range (25th percentile): ${percentile_25:.2f}
- 🟢 Upper Range (75th percentile): ${percentile_75:.2f}
- 🚀 Best Case (95th percentile): ${percentile_95:.2f}

**Probability Analysis:**
- Probability of Profit: **{prob_profit:.1f}%**
- Probability of 10%+ Gain: **{prob_gain_10:.1f}%**
- Probability of 10%+ Loss: **{prob_loss_10:.1f}%**

**Risk Assessment:**
- Price Volatility (std): ±${std_price:.2f}
- Daily Volatility: {std_return*100:.2f}%
- Annual Volatility: {std_return*np.sqrt(252)*100:.2f}%

⚠️ *Monte Carlo uses historical volatility. Past performance does not guarantee future results.*"""
        except Exception as e:
            return f"Error running Monte Carlo simulation: {e}"

    def calculate_financial_ratio(self, ratio_type, values):
        print(f"\n🔢 Calculating ratio: {ratio_type}")
        try:
            ratio_type = ratio_type.upper()
            if ratio_type == "PE":
                result = values[0] / values[1]
                return f"**P/E Ratio** = ${values[0]} / ${values[1]} = **{result:.2f}**\n\nInvestors pay ${result:.2f} for every $1 of earnings."
            elif ratio_type == "ROE":
                result = (values[0] / values[1]) * 100
                return f"**ROE** = **{result:.2f}%**"
            elif ratio_type == "DEBT_EQUITY":
                result = values[0] / values[1]
                return f"**Debt/Equity** = **{result:.2f}**"
            elif ratio_type == "GROSS_MARGIN":
                result = ((values[0] - values[1]) / values[0]) * 100
                return f"**Gross Margin** = **{result:.2f}%**"
            elif ratio_type == "NET_MARGIN":
                result = (values[0] / values[1]) * 100
                return f"**Net Margin** = **{result:.2f}%**"
            else:
                return f"Unknown ratio. Try PE, ROE, DEBT_EQUITY, GROSS_MARGIN, NET_MARGIN"
        except Exception as e:
            return f"Error: {e}"

    def calculate(self, expression):
        print(f"\n🔧 Calculating: {expression}")
        try:
            result = eval(expression)
            return f"**Result:** {expression} = **{result}**"
        except Exception as e:
            return f"Error: {e}"

    # ─────────────────────────────────────────
    # ROUTER
    # ─────────────────────────────────────────

    def decide_action(self, question):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a financial AI routing assistant. Given a question decide what action to take.

Reply with EXACTLY one of these formats:
- SEARCH_DOCS: <query>           → questions about uploaded documents
- SEARCH_WEB: <query>            → general news, current events
- STOCK: <ticker>                → stock price and metrics (e.g. STOCK: AAPL)
- HISTORY: <ticker>|<period>     → historical prices (e.g. HISTORY: AAPL|1y)
- COMPARE: <t1>|<t2>|<t3>        → compare stocks (e.g. COMPARE: AAPL|MSFT)
- FINANCIALS: <ticker>           → financial statements (e.g. FINANCIALS: AAPL)
- CALCULATE: <expression>        → math only (e.g. CALCULATE: 25 * 4.5)
- RATIO: <type>|<v1>|<v2>        → ratios (e.g. RATIO: PE|150|5)
- SHARPE: <ticker>|<period>      → Sharpe ratio (e.g. SHARPE: AAPL|1y)
- BACKTEST: <ticker>|<strategy>  → backtest (e.g. BACKTEST: AAPL|SMA)
- OPTIMIZE: <t1>|<t2>|<t3>       → portfolio optimization (e.g. OPTIMIZE: AAPL|MSFT|GOOGL)
- MONTECARLO: <ticker>|<days>    → price simulation (e.g. MONTECARLO: AAPL|30)

Examples:
"Apple stock?" → STOCK: AAPL
"AAPL history?" → HISTORY: AAPL|1y
"Compare AAPL MSFT?" → COMPARE: AAPL|MSFT
"Apple financials?" → FINANCIALS: AAPL
"Sharpe ratio AAPL?" → SHARPE: AAPL|1y
"Backtest RSI on AAPL?" → BACKTEST: AAPL|RSI
"Optimize AAPL MSFT GOOGL?" → OPTIMIZE: AAPL|MSFT|GOOGL
"Predict AAPL price?" → MONTECARLO: AAPL|30
"Monte Carlo TSLA 60 days?" → MONTECARLO: TSLA|60
"P/E price 150 EPS 5?" → RATIO: PE|150|5
"What is 25 * 4?" → CALCULATE: 25 * 4
"Market news?" → SEARCH_WEB: market news today
"Revenue in document?" → SEARCH_DOCS: revenue"""
                },
                {
                    "role": "user",
                    "content": f"History:\n{json.dumps(self.history[-4:], indent=2)}\n\nQuestion: {question}"
                }
            ]
        )
        decision = response.choices[0].message.content.strip()
        print(f"\n🤔 Decision: {decision}")
        return decision

    # ─────────────────────────────────────────
    # MAIN ASK
    # ─────────────────────────────────────────

    def ask(self, question):
        self.history.append({"role": "user", "content": question})
        decision = self.decide_action(question)

        if decision.startswith("CALCULATE:"):
            answer = self.calculate(decision.replace("CALCULATE:", "").strip())

        elif decision.startswith("STOCK:"):
            answer = self.get_stock_data(decision.replace("STOCK:", "").strip())

        elif decision.startswith("HISTORY:"):
            parts = decision.replace("HISTORY:", "").strip().split("|")
            context = self.get_historical_data(parts[0].strip(), parts[1].strip() if len(parts) > 1 else "1y")
            answer = self._llm_answer(question, context, "stock analyst")

        elif decision.startswith("COMPARE:"):
            tickers = decision.replace("COMPARE:", "").strip().split("|")
            context = self.compare_stocks(tickers)
            answer = self._llm_answer(question, context, "financial analyst")

        elif decision.startswith("FINANCIALS:"):
            answer = self.get_financials(decision.replace("FINANCIALS:", "").strip())

        elif decision.startswith("SHARPE:"):
            parts = decision.replace("SHARPE:", "").strip().split("|")
            answer = self.calculate_sharpe_ratio(parts[0].strip(), parts[1].strip() if len(parts) > 1 else "1y")

        elif decision.startswith("BACKTEST:"):
            parts = decision.replace("BACKTEST:", "").strip().split("|")
            ticker = parts[0].strip()
            strategy = parts[1].strip() if len(parts) > 1 else "SMA"
            answer = self.backtest_strategy(ticker, strategy)

        elif decision.startswith("OPTIMIZE:"):
            tickers = decision.replace("OPTIMIZE:", "").strip().split("|")
            answer = self.optimize_portfolio(tickers)

        elif decision.startswith("MONTECARLO:"):
            parts = decision.replace("MONTECARLO:", "").strip().split("|")
            ticker = parts[0].strip()
            days = int(parts[1].strip()) if len(parts) > 1 else 30
            answer = self.monte_carlo_simulation(ticker, days)

        elif decision.startswith("RATIO:"):
            parts = decision.replace("RATIO:", "").strip().split("|")
            values = [float(p.strip()) for p in parts[1:]]
            answer = self.calculate_financial_ratio(parts[0].strip(), values)

        elif decision.startswith("SEARCH_WEB:"):
            context = self.search_web(decision.replace("SEARCH_WEB:", "").strip())
            answer = self._llm_answer(question, context, "financial news analyst")

        else:
            query = decision.replace("SEARCH_DOCS:", "").strip() if decision.startswith("SEARCH_DOCS:") else question
            context = self.search_documents(query)
            answer = self._llm_answer(question, context, "financial document analyst")

        self.history.append({"role": "assistant", "content": answer})
        print(f"\n🤖 Agent: {answer}")
        return answer

    def _llm_answer(self, question, context, role="analyst"):
        system_prompt = f"""You are an expert {role} — precise, clear, and helpful.
- Start with a direct answer
- Use **bold** for key numbers
- Use bullet points for lists
- Use tables for comparisons
- End complex answers with **Summary:**
- Never make up numbers

CONTEXT:
{context}"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                *self.history
            ]
        )
        return response.choices[0].message.content