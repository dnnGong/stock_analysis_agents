from __future__ import annotations

BENCHMARK_QUESTIONS = [
    {"id": "Q01", "complexity": "easy", "category": "sector_lookup", "question": "List all semiconductor companies in the database.", "expected": "Should return semiconductor tickers/company names from local DB."},
    {"id": "Q02", "complexity": "easy", "category": "market_status", "question": "Are the US stock markets open right now?", "expected": "Should use market-status tool and report open/closed clearly."},
    {"id": "Q03", "complexity": "easy", "category": "fundamentals", "question": "What is the P/E ratio of Apple (AAPL)?", "expected": "Should return AAPL P/E ratio from tool output."},
    {"id": "Q04", "complexity": "easy", "category": "sentiment", "question": "What is the latest news sentiment for Microsoft (MSFT)?", "expected": "Should return recent sentiment label/score from news tool."},
    {"id": "Q05", "complexity": "easy", "category": "price", "question": "What is NVIDIA's stock price performance over the last month?", "expected": "Should return NVDA 1-month percentage change."},
    {"id": "Q06", "complexity": "medium", "category": "price_comparison", "question": "Compare the 1-year price performance of AAPL, MSFT, and GOOGL.", "expected": "Should provide all three returns and identify best/worst."},
    {"id": "Q07", "complexity": "medium", "category": "fundamentals", "question": "Compare the P/E ratios of AAPL, MSFT, and NVDA. Which appears most expensive?", "expected": "Should provide three P/E values and a justified comparison."},
    {"id": "Q08", "complexity": "medium", "category": "sector_price", "question": "Which energy stocks in the database had the best 6-month performance?", "expected": "Should query energy tickers then compare 6-month returns."},
    {"id": "Q09", "complexity": "medium", "category": "sentiment", "question": "What is the news sentiment for Tesla (TSLA) and how did TSLA perform over 1 month?", "expected": "Should include sentiment and 1-month return."},
    {"id": "Q10", "complexity": "medium", "category": "fundamentals", "question": "What are the 52-week high and low for JPMorgan (JPM) and Goldman Sachs (GS)?", "expected": "Should include both tickers' 52-week high/low fields."},
    {"id": "Q11", "complexity": "hard", "category": "multi_condition", "question": "Which tech stocks dropped this month but grew this year?", "expected": "Should evaluate both conditions per ticker (1mo drop and ytd growth)."},
    {"id": "Q12", "complexity": "hard", "category": "multi_condition", "question": "Which large-cap technology stocks on NASDAQ have grown more than 20% this year?", "expected": "Should filter by exchange, cap, sector, and ytd return threshold."},
    {"id": "Q13", "complexity": "hard", "category": "cross_domain", "question": "For the top 3 semiconductor stocks by 1-year return, what are their P/E ratios and current news sentiment?", "expected": "Should rank by 1-year return, then provide PE and sentiment for top 3."},
    {"id": "Q14", "complexity": "hard", "category": "cross_domain", "question": "Compare the market cap, P/E ratio, and 1-year stock performance of JPM, GS, and BAC.", "expected": "Should provide all three metrics for all three banks."},
    {"id": "Q15", "complexity": "hard", "category": "multi_condition", "question": "Which finance sector stocks are trading closer to their 52-week low than their 52-week high? Return the news sentiment for each.", "expected": "Should compute proximity-to-low condition and include sentiment for matches."},
]
