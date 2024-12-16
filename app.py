import streamlit as st
import pandas as pd
import json
from openai import OpenAI
import os
from typing import List, Dict
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_api_key():
    """Check if API key is configured"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("OpenAI API key not found. Please configure it in .env file")
        st.stop()
    return api_key

# Hardcoded master list as a string (you can paste the CSV content here)
MASTER_LIST = """Item Name,Item Type,Item Category
CE Ferulic Serum,Skincare,Product
Phloretin CF Serum,Skincare,Product
HA Intensifier,Skincare,Product
Triple Lipid Restore,Skincare,Product
Metacell Renewal B3,Skincare,Product
Retinol 0.3,Skincare,Product
Retinol 0.5,Skincare,Product
Retinol 1.0,Skincare,Product
Glycolic Renewal Cleanser,Skincare,Product
Hyaluronic Acid Serum,Skincare,Product
Vitamin C Serum,Skincare,Product
Pigment Corrector,Skincare,Product
Hydrating B5 Gel,Skincare,Product
HydraFacial Signature,Facials,Service
HydraFacial Express,Facials,Service
HydraFacial Deluxe,Facials,Service
HydraFacial Platinum,Facials,Service
LED Light Therapy,Facials,Service
Microdermabrasion,Facials,Service
Enzyme Facial,Facials,Service
Chemical Peel Basic,Facials,Service
Chemical Peel Advanced,Facials,Service
Dermaplaning,Facials,Service
Oxygen Facial,Facials,Service
Botox Forehead,Injectables,Service
Botox Glabella,Injectables,Service
Botox Crow's Feet,Injectables,Service
Botox Brow Lift,Injectables,Service
Botox Bunny Lines,Injectables,Service
Masseter Botox,Injectables,Service
Botox Nefertiti Lift,Injectables,Service
Botox Hyperhidrosis Underarm,Injectables,Service
Botox Hyperhidrosis Palms,Injectables,Service
Botox Hyperhidrosis Feet,Injectables,Service
Botox Gummy Smile,Injectables,Service
Botox Chin,Injectables,Service
Dermal Filler Lips (0.5ml),Fillers,Service
Dermal Filler Lips (1ml),Fillers,Service
Dermal Filler Nasolabial (1ml),Fillers,Service
Dermal Filler Marionette (1ml),Fillers,Service
Dermal Filler Cheeks (1ml),Fillers,Service
Dermal Filler Chin (1ml),Fillers,Service
Dermal Filler Jawline (2ml),Fillers,Service
Dermal Filler Tear Trough (1ml),Fillers,Service
Dermal Filler Nose (1ml),Fillers,Service
Dermal Filler Temple (1ml),Fillers,Service
Profhilo (2ml),Fillers,Service
Sculptra (1 Vial),Fillers,Service
Injectable Assessment,Consultation,Service
Skin Consultation,Consultation,Service
Follow Up Consultation,Consultation,Service
Treatment Planning Session,Consultation,Service
Emergency Review,Consultation,Service
Microneedling Face,Advanced Treatments,Service
Microneedling Face with PRP,Advanced Treatments,Service
Microneedling with Mesotherapy,Advanced Treatments,Service
Thread Lift PDO,Advanced Treatments,Service
Thread Lift Silhouette Soft,Advanced Treatments,Service
RF Skin Tightening,Advanced Treatments,Service
Ultrasound Skin Tightening,Advanced Treatments,Service
PRX-T33 Treatment,Advanced Treatments,Service
Mesotherapy Face,Advanced Treatments,Service
Mesotherapy Hair,Advanced Treatments,Service
Fat Dissolving Injection,Advanced Treatments,Service
Plasma Pen Treatment,Advanced Treatments,Service
Body Fat Reduction Coolsculpting,Body Treatments,Service
Body Fat Reduction Deoxycholic Acid,Body Treatments,Service
EMSculpt Neo,Body Treatments,Service
Cellulite Treatment,Body Treatments,Service
Body Contouring RF,Body Treatments,Service
Laser Hair Removal Small Area,Body Treatments,Service
Laser Hair Removal Medium Area,Body Treatments,Service
Laser Hair Removal Large Area,Body Treatments,Service
SPF 30,Retail Products,Product
SPF 50,Retail Products,Product
Post Procedure Kit,Retail Products,Product
Gentle Cleanser,Retail Products,Product
Moisturizing Cream,Retail Products,Product
Eye Cream,Retail Products,Product
Neck Cream,Retail Products,Product
Lip Care,Retail Products,Product
Scar Treatment Gel,Retail Products,Product
Redness Relief Cream,Retail Products,Product
Acne Treatment Serum,Retail Products,Product
Hair Growth Serum,Retail Products,Product
Brightening Cream,Retail Products,Product"""


class LLMItemMapper:
    def __init__(self, master_df: pd.DataFrame):
        self.master_df = master_df
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.mapping_cache = {}
        
    def create_system_prompt(self) -> str:
        return f"""You are an expert in medical aesthetics product naming standardization.
Your task is to match clinic items to our master catalog while considering context and industry knowledge.

Master list examples:
{self.master_df.head(3).to_string()}

Respond with only a JSON object in this exact format:
{{
    "matched_name": "exact name from master list",
    "confidence": 0.XX,
    "reasoning": "brief explanation"
}}"""

    def create_item_prompt(self, item_name: str, item_type: str, item_category: str) -> str:
        return f"""Find the best match for:
Item Name: {item_name}
Type: {item_type}
Category: {item_category}

Remember to respond with only a JSON object."""

    def extract_json_from_response(self, text: str) -> Dict:
        """Extract JSON from the response text."""
        try:
            # Find the first { and last } in the text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                return json.loads(json_str)
            raise ValueError("No JSON object found in response")
        except Exception as e:
            raise ValueError(f"Failed to parse response: {str(e)}")

    def map_item(self, item_name: str, item_type: str, item_category: str) -> Dict:
        cache_key = f"{item_name}|{item_type}|{item_category}"
        if cache_key in self.mapping_cache:
            return self.mapping_cache[cache_key]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" if preferred
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": self.create_item_prompt(item_name, item_type, item_category)}
                ],
                temperature=0.3
            )
            
            # Extract JSON from response text
            result = self.extract_json_from_response(response.choices[0].message.content)
            result['original_name'] = item_name
            
            self.mapping_cache[cache_key] = result
            return result
            
        except Exception as e:
            st.error(f"Error processing item {item_name}: {str(e)}")
            return {
                'original_name': item_name,
                'matched_name': None,
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)}"
            }



def main():
    # Page config
    st.set_page_config(
        page_title="CPP Europe Item Name Standardization",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Headers
    st.markdown('<h1 class="main-header">AI Engine for Item Standardization</h1>', unsafe_allow_html=True)
    st.markdown("""
        <p class="sub-header">
        Base Input is a curated master list that our LLM is trained on.<br>
        User inputs sample data from transactions (name, type, category) and output is a suggested name 
        for each row with confidence level and reasoning.
        </p>
        """, unsafe_allow_html=True)
    
    # Load master list
    master_df = pd.DataFrame([x.split(',') for x in MASTER_LIST.split('\n')[1:]], 
                           columns=MASTER_LIST.split('\n')[0].split(','))
    
    # Master list preview with column configuration
    st.markdown("### Master List Preview")
    st.dataframe(
        master_df,
        column_config={
            "Item Name": st.column_config.TextColumn("Item Name", width="medium"),
            "Item Type": st.column_config.TextColumn("Type", width="small"),
            "Item Category": st.column_config.TextColumn("Category", width="small"),
            "Brand": st.column_config.TextColumn("Brand", width="small")
        },
        hide_index=True
    )
    
    # Input section
    st.markdown("### Enter Items to Standardize")
    st.caption("CSV format: Item Name, Item Type, Item Category")
    sample_text = st.text_area(
        label="Paste sample data",
        value="""C+E Fer,Product,Face Serums
Phloretin CF (30ml),Product,Face Serum
HA Int,Product,Face Treatment
Triple lipid restore,Product,Face Cream
AGE eye,Product,Eye Products
Disc Defense serum,Product,Pigmentation""",
        height=200
    )
    
    # Processing
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Map Items", type="primary", use_container_width=True):
            with st.spinner("Processing items..."):
                # Parse input data
                sample_data = []
                for line in sample_text.split('\n'):
                    if line.strip():
                        try:
                            name, type_, category = [x.strip() for x in line.split(',')]
                            sample_data.append({
                                'Item Name': name,
                                'Item Type': type_,
                                'Item Category': category
                            })
                        except ValueError:
                            st.error(f"Invalid line format: {line}")
                            continue
                
                # Process items
                sample_df = pd.DataFrame(sample_data)
                mapper = LLMItemMapper(master_df)
                results = []
                
                # Progress tracking
                progress_bar = st.progress(0)
                for idx, row in enumerate(sample_df.iterrows()):
                    result = mapper.map_item(
                        row[1]['Item Name'],
                        row[1]['Item Type'],
                        row[1]['Item Category']
                    )
                    results.append(result)
                    progress_bar.progress((idx + 1) / len(sample_df))
                
                # Display results
                st.markdown("### Results")
                display_df = pd.DataFrame({
                    'Original Name': [r['original_name'] for r in results],
                    'Matched Name': [r['matched_name'] for r in results],
                    'Confidence': [r['confidence'] for r in results],
                    'Reasoning': [r['reasoning'] for r in results]
                })
                
                # Style results
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
                
                st.dataframe(styled_df, hide_index=True)
                
                # Download option
                if len(results) > 0:
                    st.download_button(
                        label="Download Results CSV",
                        data=display_df.to_csv(index=False),
                        file_name="mapping_results.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()