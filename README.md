# CPP Product Name Standardization

A Streamlit application for standardizing product names using GPT-4.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cpp_demo_v1.git
cd cpp_demo_v1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
cp .env.example .env
```

5. Edit `.env` and add your OpenAI API key

6. Run the application:
```bash
streamlit run app.py
```

## Usage

1. The application shows the master list of standardized product names
2. Paste your data in CSV format: Item Name, Item Type, Item Category
3. Click "Map Items" to process
4. Download results as CSV

## Development

- Master list is stored in the `MASTER_LIST` variable in `app.py`
- Environment variables are loaded from `.env`
- The application uses GPT-4 for intelligent matching

## Security

- Never commit the `.env` file
- Always use environment variables for secrets
- The repository includes push protection for sensitive data