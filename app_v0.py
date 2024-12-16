import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import io
import numpy as np

class SimpleItemMapper:
    def __init__(self, master_df):
        self.master_df = master_df
        self.mapping_cache = {}
        
    def preprocess_name(self, name: str) -> str:
        return name.lower().strip().replace("  ", " ")
        
    def calculate_similarity(self, name1: str, name2: str) -> float:
        return fuzz.ratio(self.preprocess_name(name1), self.preprocess_name(name2)) / 100.0
        
    def map_item(self, item_name: str, item_type: str, item_category: str) -> dict:
        if item_name in self.mapping_cache:
            return self.mapping_cache[item_name]
            
        matches = []
        for _, master_item in self.master_df.iterrows():
            base_similarity = self.calculate_similarity(item_name, master_item['standard_name'])
            
            category_match = (master_item['category'].lower() == item_category.lower())
            type_match = (master_item['type'].lower() == item_type.lower())
            
            confidence = base_similarity
            if category_match:
                confidence += 0.1
            if type_match:
                confidence += 0.1
                
            matches.append({
                'suggested_name': master_item['standard_name'],
                'confidence': min(confidence, 1.0),
                'original_name': item_name,
                'category_match': category_match,
                'type_match': type_match,
                'master_category': master_item['category']
            })
            
        best_match = max(matches, key=lambda x: x['confidence'])
        self.mapping_cache[item_name] = best_match
        return best_match

def main():
    st.set_page_config(page_title="CPP Item Name Standardization", layout="wide")
    
    st.title("CPP Europe Item Name Standardization")
    st.write("Upload master list and input sample items to standardize names")
    
    # File upload for master list
    uploaded_file = st.file_uploader("Upload Master List CSV", type=['csv'])
    
    if uploaded_file is not None:
        master_df = pd.read_csv(uploaded_file)
        st.write("Master List Preview:")
        st.dataframe(master_df.head())
        
        # Text area for sample data input
        st.write("Enter sample items (CSV format: Item Name, Item Type, Item Category)")
        sample_text = st.text_area(
            "Paste 3-4 rows of sample data",
            "C E Ferulic, Product, Skincare\nHydraFacial Platinum, Service, Facials\nBotox - Forehead, Service, Injectables",
            height=100
        )
        
        if st.button("Map Items"):
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
            mapper = SimpleItemMapper(master_df)
            results = []
            
            for _, row in sample_df.iterrows():
                result = mapper.map_item(
                    row['Item Name'],
                    row['Item Type'],
                    row['Item Category']
                )
                results.append(result)
            
            # Display results
            results_df = pd.DataFrame(results)
            
            # Create two separate dataframes - one for display and one for styling
            display_df = pd.DataFrame({
                'Original Name': results_df['original_name'],
                'Suggested Name': results_df['suggested_name'],
                'Confidence': results_df['confidence'].apply(lambda x: f"{x:.1%}"),
                'Master Category': results_df['master_category']
            })
            
            # Create a numeric dataframe for styling
            style_df = display_df.copy()
            style_df['Confidence'] = results_df['confidence']  # Use raw numeric values
            
            # Apply styling and show display values
            st.dataframe(
                style_df.style.background_gradient(
                    subset=['Confidence'],
                    cmap='RdYlGn'
                ).format({
                    'Confidence': '{:.1%}'
                })
            )
            
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