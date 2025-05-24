# Web Scraping Agent with LangChain

A powerful web scraping agent that uses LangChain to crawl websites, extract structured content, and create a vector database for semantic search. This tool is particularly useful for creating knowledge bases from websites and performing intelligent searches on the scraped content.

## Features

- **Recursive Web Crawling**: Automatically follows and scrapes linked pages up to a specified depth
- **Structured Content Extraction**:
  - Headers (H1, H2, H3)
  - Paragraphs
  - Lists (ordered and unordered)
  - Tables
  - Images with alt text
  - Links
  - Meta descriptions
- **Vector Database Integration**:
  - Support for FAISS and Chroma vector stores
  - Semantic search capabilities
  - Efficient text chunking
- **Smart Content Processing**:
  - Automatic content chunking
  - Text normalization
  - Duplicate prevention
  - URL validation and normalization
- **Robust Error Handling**:
  - Automatic retries for failed requests
  - Rate limiting
  - Comprehensive error reporting
- **Data Export**:
  - JSON export of all scraped content
  - Structured data format
  - Easy integration with other tools

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd web-scraper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Basic usage:
```python
from web_scraper_agent import WebScraperAgent

# Initialize the agent
agent = WebScraperAgent(
    url="https://example.com",
    vector_store_type="faiss",  # or "chroma"
    max_depth=2  # depth of crawling
)

# Start scraping and create vector store
vector_store = agent.process_website()

# Perform semantic search
results = vector_store.similarity_search("Your question here")
```

2. Customize scraping depth:
```python
# Scrape only the main page
agent = WebScraperAgent(url="https://example.com", max_depth=1)

# Scrape main page and linked pages
agent = WebScraperAgent(url="https://example.com", max_depth=2)

# Deep crawling
agent = WebScraperAgent(url="https://example.com", max_depth=3)
```

## Configuration

The agent can be configured with the following parameters:

- `url`: The starting URL for scraping
- `vector_store_type`: Choose between "faiss" or "chroma"
- `max_depth`: Maximum depth for recursive crawling
- `chunk_size`: Size of text chunks for vector store (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

## Output

The scraper generates:
1. A detailed console output showing:
   - Crawling statistics
   - Content structure
   - Extracted information
2. A JSON file containing all scraped data
3. A vector store for semantic search

## Dependencies

- langchain
- beautifulsoup4
- requests
- faiss-cpu
- chromadb
- python-dotenv
- tiktoken
- sentence-transformers

## Best Practices

1. **Rate Limiting**: The scraper includes built-in delays between requests to avoid overwhelming servers
2. **Error Handling**: Automatic retries and comprehensive error reporting
3. **Content Validation**: Checks for valid content before processing
4. **Memory Management**: Efficient handling of large websites
5. **Data Organization**: Structured storage of scraped content

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.