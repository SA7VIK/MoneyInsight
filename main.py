from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import groq
import os
from dotenv import load_dotenv
from typing import Optional
import json
import re

# Load environment variables
load_dotenv()

app = FastAPI(title="Investment Allocation API")

# Initialize Groq client with error handling
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    client = groq.Client(api_key=api_key)
except Exception as e:
    print(f"Error initializing Groq client: {str(e)}")
    client = None

# Hardcoded market trends
MARKET_TRENDS = {
    "NIFTY 50": "Expected to increase by 12-15% in the next year due to strong economic growth and corporate earnings",
    "Gold": "Expected to increase by 8-10% in the next year due to global economic uncertainty and inflation concerns",
    "Bonds": "Expected to yield 6-7% in the next year with moderate risk",
    "Real Estate": "Expected to appreciate by 5-7% in major metropolitan areas",
    "Cryptocurrency": "High volatility expected with potential 20-30% gains but significant risk",
    "Fixed Deposits": "Offering 6-7% returns with minimal risk",
    "International Markets": "Expected to grow by 10-12% with focus on US and European markets"
}

class InvestmentRequest(BaseModel):
    age: int
    investment_amount: float
    market_forecast: Optional[str] = None

class InvestmentResponse(BaseModel):
    allocation_percentage: dict
    recommendation: str

def extract_json_from_text(text):
    # Try to find JSON-like structure in the text
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, text)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If no valid JSON found, try to parse the text into a structured format
    try:
        # Extract allocation percentages
        allocation = {}
        for asset in MARKET_TRENDS.keys():
            # Look for percentages associated with each asset
            pattern = rf"{asset}.*?(\d+)%"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                allocation[asset] = int(match.group(1))
        
        # If we found allocations, create a response
        if allocation:
            return {
                "allocation": allocation,
                "explanation": text
            }
    except Exception:
        pass
    
    # If all parsing attempts fail, return a default response
    return {
        "allocation": {"Cash": 100},
        "explanation": "Unable to parse specific allocations. Please review the full response for details."
    }

@app.post("/allocate-investment", response_model=InvestmentResponse)
async def allocate_investment(request: InvestmentRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Groq client not initialized properly")
        
    try:
        # Prepare the market trends section
        market_trends_text = "\n".join([f"- {asset}: {trend}" for asset, trend in MARKET_TRENDS.items()])
        
        # Prepare the prompt for Groq
        prompt = f"""
        As a financial advisor, analyze the following investment scenario and provide allocation recommendations:
        
        Investor Age: {request.age}
        Investment Amount: ${request.investment_amount:,.2f}
        
        Current Market Trends and Forecasts (Next 1 Year):
        {market_trends_text}
        
        Additional Market Context:
        {request.market_forecast if request.market_forecast else "No additional market forecast provided"}
        
        Please provide:
        1. A detailed allocation percentage breakdown across different asset classes (stocks, bonds, cash, etc.)
        2. A brief explanation for the allocation strategy, taking into account the provided market trends
        
        Format your response as a JSON object with 'allocation' and 'explanation' fields.
        The allocation should be a dictionary with asset classes as keys and percentages as values.
        Example format:
        {{
            "allocation": {{
                "NIFTY 50": 40,
                "Gold": 20,
                "Bonds": 25,
                "Cash": 15
            }},
            "explanation": "Detailed explanation here"
        }}
        
        IMPORTANT: Your response must be a valid JSON object with the exact structure shown above.
        """

        # Get response from Groq
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional financial advisor with expertise in investment allocation and portfolio management. Consider the investor's age, risk tolerance, and market trends when making recommendations. Always respond with a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        # Extract the response
        response_text = completion.choices[0].message.content

        # Parse the response to get allocation percentages
        try:
            allocation_data = extract_json_from_text(response_text)
            return InvestmentResponse(
                allocation_percentage=allocation_data.get('allocation', {}),
                recommendation=allocation_data.get('explanation', '')
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing LLM response: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Investment Allocation API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 