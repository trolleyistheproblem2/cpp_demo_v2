import streamlit as st
import pandas as pd
import json
from openai import OpenAI
from typing import List, Dict, Tuple
import time
from tqdm import tqdm

class LLMItemMapper:
    def __init__(self, master_df: pd.DataFrame, api_key: str):
        """
        Initialize the LLM-powered item mapper.
        
        Args:
            master_df (pd.DataFrame): Master list of standardized items
            api_key (str): OpenAI API key
        """
        self.master_df = master_df
        self.client = OpenAI(api_key=api_key)
        self.mapping_cache = {}
        
    def create_system_prompt(self) -> str:
        """Create a detailed system prompt with the master list context."""
        categories = self.master_df['category'].unique().tolist()
        types = self.master_df['type'].unique().tolist()
        
        return f"""You are an expert in medical aesthetics product and service naming standardization.
Your task is to match clinic items to our master catalog while considering context and industry knowledge.

Available Categories: {categories}
Available Types: {types}

For each item, you must respond in valid JSON format with these exact fields:
- matched_name (string): exact name from master list
- confidence (number): between 0 and 1
- reasoning (string): brief explanation of the match
- alternative_matches (array of strings): other possible matches if confidence < 0.9

Master list examples:
{self.master_df.head(3).to_string()}"""

    def create_item_prompt(self, item_name: str, item_type: str, item_category: str) -> str:
        """Create a prompt for a specific item."""
        return f"""Analyze this item and find the best match in our master list:
Item Name: {item_name}
Type: {item_type}
Category: {item_category}

Consider industry standard abbreviations, common misspellings, and regional variations.
Your response MUST be in this exact JSON format:
{{
    "matched_name": "exact name from master list",
    "confidence": 0.95,
    "reasoning": "brief explanation of why this is a match",
    "alternative_matches": ["other possible match 1", "other possible match 2"]
}}"""

    def process_llm_response(self, response: str) -> Dict:
        """Process and validate LLM response."""
        try:
            # Extract JSON from the response text
            # Look for content between first { and last }
            json_str = response[response.find('{'):response.rfind('}')+1]
            result = json.loads(json_str)
            
            required_fields = ['matched_name', 'confidence', 'reasoning']
            if not all(field in result for field in required_fields):
                raise ValueError("Missing required fields in LLM response")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Error parsing LLM response: {str(e)}\nResponse was: {response}")
            return {
                'matched_name': None,
                'confidence': 0.0,
                'reasoning': f"Failed to parse LLM response: {str(e)}"
            }

    def map_item(self, item_name: str, item_type: str, item_category: str) -> Dict:
        """
        Map an item to the master list using LLM.
        
        Args:
            item_name (str): Name of the item to match
            item_type (str): Type of the item (Product/Service)
            item_category (str): Category of the item
            
        Returns:
            Dict containing match results
        """
        # Check cache first
        cache_key = f"{item_name}|{item_type}|{item_category}"
        if cache_key in self.mapping_cache:
            return self.mapping_cache[cache_key]

        try:
            # Call OpenAI API without response_format parameter
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": self.create_item_prompt(item_name, item_type, item_category)}
                ],
                temperature=0.3  # Lower temperature for more consistent results
            )
            
            result = self.process_llm_response(response.choices[0].message.content)
            
            # Add original item details to result
            result['original_name'] = item_name
            result['original_type'] = item_type
            result['original_category'] = item_category
            
            # Cache the result
            self.mapping_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            st.error(f"Error in LLM processing: {str(e)}")
            return {
                'original_name': item_name,
                'matched_name': None,
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)}"
            }

def main():
    st.set_page_config(page_title="CPP Europe Item Name Standardization (LLM Enhanced)", layout="wide")
    
    st.title("CPP Europe Item Name Standardization")
    st.write("Enhanced with GPT-4 for intelligent matching")
    
    # API Key input
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    # File upload for master list
    uploaded_file = st.file_uploader("Upload Master List CSV", type=['csv'])
    
    if uploaded_file is not None and api_key:
        master_df = pd.read_csv(uploaded_file)
        st.write("Master List Preview:")
        st.dataframe(master_df.head())
        
        # Text area for sample data input
        st.write("Enter sample items (CSV format: Item Name, Item Type, Item Category)")
        sample_text = st.text_area(
            "Paste sample data",
            "C E Ferulic, Product, Skincare\nHydraFacial Platinum, Service, Facials",
            height=100
        )
        
        if st.button("Map Items"):
            with st.spinner("Processing items with GPT-4..."):
                # Convert sample text to DataFrame
                sample_data = []
                for line in sample_text.split('\n'):
                    if line.strip():
                        name, type_, category = [x.strip() for x in line.split(',')]
                        sample_data.append({
                            'Item Name': name,
                            'Item Type': type_,
                            'Item Category': category
                        })
                
                sample_df = pd.DataFrame(sample_data)
                
                # Initialize mapper and process items
                mapper = LLMItemMapper(master_df, api_key)
                results = []
                
                # Process items with progress bar
                progress_bar = st.progress(0)
                for idx, row in enumerate(sample_df.iterrows()):
                    result = mapper.map_item(
                        row[1]['Item Name'],
                        row[1]['Item Type'],
                        row[1]['Item Category']
                    )
                    results.append(result)
                    progress_bar.progress((idx + 1) / len(sample_df))
                
                # Display results with styling
                st.subheader("Mapping Results")
                
                # Create a styled view of the results
                display_df = pd.DataFrame({
                    'Original Name': [r['original_name'] for r in results],
                    'Matched Name': [r['matched_name'] for r in results],
                    'Confidence': [r['confidence'] for r in results],
                    'Reasoning': [r['reasoning'] for r in results]
                })
                
                # Style the DataFrame
                def get_color(confidence):
                    if confidence >= 0.9:
                        return 'background-color: #90EE90'
                    elif confidence >= 0.7:
                        return 'background-color: #FFFFE0'
                    return 'background-color: #FFB6C1'
                
                styled_df = display_df.style.apply(
                    lambda x: [get_color(v) if i == 2 else '' for i, v in enumerate(x)], 
                    axis=1
                ).format({
                    'Confidence': '{:.1%}'
                })
                
                st.dataframe(styled_df)
                
                # Download results button
                if len(results) > 0:
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="mapping_results.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()